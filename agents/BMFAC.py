import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import os,sys
import wandb
sys.path.append(os.path.abspath('../..'))

import matplotlib.pyplot as plt
from agents.models import Actor, MFCritic, BMF_actor, BMF_critic, BMF_actor_1
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from agents.utils import get_action_info
from datetime import datetime
from tensorboardX import SummaryWriter
from env.evaluation import save_parameters

from omegaconf import OmegaConf

torch.autograd.set_detect_anomaly(True)


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

'''
Bi-level: government and households 
mean field: 
    - gov has a actor and a critic;  pi(og), Q(og, ag, bar{ah})
    - households share a actor and a critic. pi(at | ot, ag, bar{a})  Q(oh, ag, ah^i, bar{ah^-i} )
'''
class BMFAC_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        self.eval_env = copy.copy(envs)
        # start to build the network.
        gov_obs_dim = self.envs.government.observation_space.shape[0]
        gov_action_dim = self.envs.government.action_space.shape[0]
        house_obs_dim = self.envs.households.observation_space.shape[0]
        house_action_dim = self.envs.households.action_space.shape[1]

        self.gov_actor = Actor(gov_obs_dim, gov_action_dim, self.args.hidden_size, self.args.log_std_min, self.args.log_std_max)
        # self.house_actor = BMF_actor(house_obs_dim, gov_action_dim, house_action_dim, self.args.n_households, self.args.log_std_min, self.args.log_std_max)
        self.house_actor = BMF_actor_1(house_obs_dim, gov_action_dim, house_action_dim, self.args.n_households, self.args.log_std_min, self.args.log_std_max)
        self.gov_critic = MFCritic(gov_obs_dim, self.args.hidden_size, gov_action_dim, house_action_dim*2)
        self.target_gov_qf = copy.deepcopy(self.gov_critic)
        self.house_critic = BMF_critic(house_obs_dim, gov_action_dim, house_action_dim, self.args.hidden_size, self.args.n_households)
        self.target_house_qf = copy.deepcopy(self.house_critic)

        # if use the cuda...
        if self.args.cuda:
            self.gov_actor.cuda()
            self.house_actor.cuda()
            self.gov_critic.cuda()
            self.house_critic.cuda()
            self.target_gov_qf.cuda()
            self.target_house_qf.cuda()

        # define the optimizer...
        self.gov_critic_optim = torch.optim.Adam(self.gov_critic.parameters(), lr=self.args.q_lr)
        self.house_critc_optim = torch.optim.Adam(self.house_critic.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.gov_actor_optim = torch.optim.Adam(self.gov_actor.parameters(), lr=self.args.p_lr)
        self.house_actor_optim = torch.optim.Adam(self.house_actor.parameters(), lr=self.args.p_lr)

        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)

        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0][0]

        self.model_path, _ = make_logpath(algo="bmfac",n=self.args.n_households)
        save_args(path=self.model_path, args=self.args)

        self._wealth_buffer = []                  # holds list of rows (each row is a list: [epoch, w0, w1, ...])
        self._wealth_buffer_interval = 50
        self._income_pre_buffer = []    # gross (market) income
        self._income_post_buffer = []   # post-tax income

        self._income_buffer_interval = 50
        
        self.fix_gov = True
        self.wandb = False
        #self.wandb = False
        cfg_for_wandb = OmegaConf.to_container(self.args, resolve=True)
        if self.wandb:
            wandb.init(
                config=cfg_for_wandb,
                project="",
                entity="",
                name=self.model_path.parent.parent.name+ "-"+ self.model_path.name +'  n='+ str(self.args.n_households),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )

    def _flush_wealth_buffer(self):
        """Append buffered rows to CSV in one write. Creates folders & header if missing."""
        if len(self._wealth_buffer) == 0:
            return

        out_dir = self.model_path / "households"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "wealth_epochs.csv"

        # write header if file doesn't exist
        if not csv_path.exists():
            with open(csv_path, "w") as f:
                header = "epoch," + ",".join([f"h{i}" for i in range(len(self._wealth_buffer[0]) - 1)])
                f.write(header + "\n")

        # append all buffered rows at once
        with open(csv_path, "a") as f:
            for row in self._wealth_buffer:
                f.write(",".join([str(x) for x in row]) + "\n")

        # clear buffer
        self._wealth_buffer = []


    def _flush_income_pre_buffer(self):
        if len(self._income_pre_buffer) == 0:
            return

        out_dir = self.model_path / "households"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "income_pre_epochs.csv"

        if not csv_path.exists():
            with open(csv_path, "w") as f:
                header = "epoch," + ",".join(
                    [f"h{i}" for i in range(len(self._income_pre_buffer[0]) - 1)]
                )
                f.write(header + "\n")

        with open(csv_path, "a") as f:
            for row in self._income_pre_buffer:
                f.write(",".join(str(x) for x in row) + "\n")

        self._income_pre_buffer = []


    def _flush_income_post_buffer(self):
        if len(self._income_post_buffer) == 0:
            return

        out_dir = self.model_path / "households"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "income_post_epochs.csv"

        if not csv_path.exists():
            with open(csv_path, "w") as f:
                header = "epoch," + ",".join(
                    [f"h{i}" for i in range(len(self._income_post_buffer[0]) - 1)]
                )
                f.write(header + "\n")

        with open(csv_path, "a") as f:
            for row in self._income_post_buffer:
                f.write(",".join(str(x) for x in row) + "\n")

        self._income_post_buffer = []

    def house_action_initialize(self):
        action = np.random.uniform(low=-self.hou_action_max, high=self.hou_action_max, size=(1, self.args.n_households, self.envs.households.action_space.shape[1]))
        mean_action = np.mean(action, axis=1)
        return mean_action

    def multiple_households_mean_action(self, actions=None):
        if actions is None:
            action = np.random.uniform(low=-self.hou_action_max, high=self.hou_action_max,
                                       size=(1, self.args.n_households, self.envs.households.action_space.shape[1]))
            mean_action = np.mean(action, axis=1)
            return np.hstack((mean_action, mean_action))
        else:
            wealth = self.envs.households.at_next
            sorted_wealth_index = sorted(range(len(wealth)), key=lambda k: wealth[k], reverse=True)
            top10_wealth_index = sorted_wealth_index[:int(0.1 * self.args.n_households)]
            bottom50_wealth_index = sorted_wealth_index[int(0.5 * self.args.n_households):]
            top10_action = actions[top10_wealth_index]
            bot50_action = actions[bottom50_wealth_index]
            # return top10_action, bot50_action
            return np.hstack((np.mean(top10_action,axis=0), np.mean(bot50_action,axis=0)))[np.newaxis,:]
    def get_tensor_mean_action(self, actions, wealth):
        sorted_wealth_index = torch.sort(wealth[:, :, 0], dim=1)[1]
        top10_wealth_index = sorted_wealth_index[:, :int(0.1 * self.args.n_households)]
        bottom50_wealth_index = sorted_wealth_index[:, int(0.5 * self.args.n_households):]
        top10_action = actions.gather(1, top10_wealth_index.unsqueeze(2).expand(-1, -1, self.envs.households.action_space.shape[1]))
        bot50_action = actions.gather(1, bottom50_wealth_index.unsqueeze(2).expand(-1, -1, self.envs.households.action_space.shape[1]))

        return torch.cat((torch.mean(top10_action, dim=1), torch.mean(bot50_action, dim=1)), 1)
    def observation_wrapper(self, global_obs, private_obs):
        # global
        global_obs[0] /= 1e7
        global_obs[1] /= 1e5
        global_obs[3] /= 1e5
        global_obs[4] /= 1e5
        private_obs[:, 1] /= 1e5
        return global_obs, private_obs

    def learn(self):
        update_freq = self.args.update_freq
        # for loop
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy=self.args.init_exploration_policy)
        # reset the environment
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        # past_mean_house_action = self.multiple_households_mean_action()
        gov_rew = []
        house_rew = []
        epochs = []
        agent_list = ["households", "government"]
        update_index = 0
        max_sw = 0

        # for epoch in range(1):
        for epoch in range(self.args.n_epochs):
            if epoch == self.args.n_epochs - 1:
                self.result_evaluation = True
            else:
                self.result_evaluation = False
            # self.save_true = False
            if epoch % update_freq == 0:
                update_index = 1 - update_index
            update_agent = agent_list[update_index]
            print("update_agent:", update_agent)

            # for each epoch, it will reset the environment
            for t in range(self.args.epoch_length):
                # start to collect samples
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                gov_pi = self.gov_actor(global_obs_tensor)
                gov_action = get_action_info(gov_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                hou_pi = self.house_actor(global_obs_tensor, private_obs_tensor, gov_action)
                hou_action = get_action_info(hou_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                gov_action = gov_action.cpu().numpy()[0]
                hou_action = hou_action.cpu().numpy()[0]

                past_mean_house_action = self.multiple_households_mean_action(hou_action)[0]
                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: hou_action}
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                # store the episodes
                self.buffer.mf_add(global_obs, private_obs, gov_action, hou_action, past_mean_house_action, gov_reward, house_reward,
                                next_global_obs, next_private_obs, float(done))
                # past_mean_house_action = self.multiple_households_mean_action(hou_action)
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()
                    global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
                # update frequency
                if t % 50 == 0:
                    gov_actor_loss, gov_critic_loss, house_actor_loss, house_critic_loss = self._update_network(update_agent=update_agent)
                    # update the target network
                    if global_timesteps % self.args.target_update_interval == 0:
                        self._update_target_network(self.target_gov_qf, self.gov_critic)
                        self._update_target_network(self.target_house_qf, self.house_critic)
                global_timesteps += 1
            # print the log information
            if epoch % self.args.display_interval == 0:
                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years, avg_ubi_prop_gdp, avg_cons_std_overall, avg_cons_cv_overall, avg_cons_std_top10, \
                    avg_cons_cv_top10, avg_cons_std_bot50, avg_cons_cv_bot50, avg_cons_lag1_overall, avg_cons_lag1_top10, avg_cons_lag1_bot50, avg_debt, avg_debt_to_gdp, \
                         avg_mean_work_hours, avg_participation, avg_mean_work_top10, avg_mean_work_bot50,  avg_mean_saving_prop, avg_mean_saving_top10, avg_mean_saving_bot50, avg_penalty_freq, \
                             avg_cons_cv_overall_inc, avg_cons_cv_top10_inc, avg_cons_cv_bot50_inc, tax_revenue_pct_gdp, avg_tax_share_top10, avg_tax_share_bot50, \
                                avg_etr_overall, avg_etr_top10, avg_etr_bot50, avg_income_tax_rate_overall, avg_income_tax_rate_top10, avg_income_tax_rate_bot50, avg_wealth_tax_rate_overall, avg_wealth_tax_rate_top10, avg_wealth_tax_rate_bot50  = self._evaluate_agent()
            
                tax_minus_ubi_pct_gdp = tax_revenue_pct_gdp - avg_ubi_prop_gdp
                ubi_per_hh = avg_ubi_prop_gdp * avg_gdp
                ubi_share_of_wealth = ubi_per_hh / avg_mean_wealth
                # store rewards and step
                
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)

                wealth = self.envs.households.at_next.flatten()

                # prepare row: [epoch, w0, w1, ...]
                row = [int(epoch)] + [float(w) for w in wealth]

                # append to in-memory buffer
                self._wealth_buffer.append(row)

                # flush when buffer reaches capacity (or you can also flush on last epoch below)
                if len(self._wealth_buffer) >= self._wealth_buffer_interval:
                    self._flush_wealth_buffer()
                
                income_pre = self.envs.income.flatten()
                self._income_pre_buffer.append([int(epoch)] + income_pre.tolist())

                # post-tax income (used for income_gini)
                income_post = self.envs.post_income.flatten()
                self._income_post_buffer.append([int(epoch)] + income_post.tolist())

                # flush
                if len(self._wealth_buffer) >= self._wealth_buffer_interval:
                    self._flush_wealth_buffer()

                if len(self._income_pre_buffer) >= self._income_buffer_interval:
                    self._flush_income_pre_buffer()

                if len(self._income_post_buffer) >= self._income_buffer_interval:
                    self._flush_income_post_buffer()



                if self.wandb:
                    
                    percentiles = self.envs.stacked_data(self.eval_env.households.at_next.flatten())

                    decile_ranges = [
                        (90, 100),  # richest 10%
                        (80, 90),
                        (70, 80),
                        (60, 70),
                        (50, 60),
                        (40, 50),
                        (30, 40),
                        (20, 30),
                        (10, 20),
                        (0, 10)     # poorest 10%
                    ]

                    # collect everything into one dict, then log once
                    log_dict = {}

                    # deciles
                    for (lo, hi), value in zip(decile_ranges, percentiles):
                        log_dict[f"wealth_decile_{lo}_{hi}"] = float(value)

                    # add all the other metrics into the same dict (rename keys if you want consistent names)
                    log_dict.update({
                        "mean_households_utility": mean_house_rewards,
                        "goverment_utility": mean_gov_rewards,
                        "years": years,
                        "wealth_gini": avg_wealth_gini,
                        "income_gini": avg_income_gini,
                        "per_household_gdp": avg_gdp,
                        "tax_per_household": avg_mean_tax,
                        "tax_revenue_pct_gdp": tax_revenue_pct_gdp,
                        "post_income_per_household": avg_mean_post_income,
                        "wealth_per_household": avg_mean_wealth,
                        "government_actor_loss": gov_actor_loss,
                        "government_critic_loss": gov_critic_loss,
                        "households_actor_loss": house_actor_loss,
                        "households_critic_loss": house_critic_loss,
                        "ubi_prop_gdp": avg_ubi_prop_gdp,
                        "tax_minus_ubi_pct_gdp": tax_minus_ubi_pct_gdp,
                        "ubi_share_of_wealth": ubi_share_of_wealth,
                        "consumption_standard_deviation_overall": avg_cons_std_overall,
                        "consumption_coeff_variation_overall": avg_cons_cv_overall,
                        "consumption_standard_deviation_top10": avg_cons_std_top10,
                        "consumption_coeff_variation_top10": avg_cons_cv_top10,
                        "consumption_standard_deviation_bot50": avg_cons_std_bot50,
                        "consumption_coeff_variation_bot50": avg_cons_cv_bot50,
                        "pearson_corr_consumption_lag1_overall": avg_cons_lag1_overall,
                        "pearson_corr_consumption_lag1_top10": avg_cons_lag1_top10,
                        "pearson_corr_consumption_lag1_bot50": avg_cons_lag1_bot50,
                        "nominal_debt": avg_debt,
                        "debt_to_gdp_ratio": avg_debt_to_gdp,
                        "mean_working_hours": avg_mean_work_hours,
                        "labour_participation_rate": avg_participation,
                        "top10_mean_working_hours": avg_mean_work_top10,
                        "bottom50_mean_working_hours": avg_mean_work_bot50,
                        "mean_saving_propensity_overall": avg_mean_saving_prop,
                        "mean_saving_propensity_top10": avg_mean_saving_top10,
                        "mean_saving_propensity_bot50": avg_mean_saving_bot50,
                        "excess_penalty_years_pct": avg_penalty_freq,
                        "avg_cons_cv_overall_inc": avg_cons_cv_overall_inc,
                        "avg_cons_cv_top10_inc": avg_cons_cv_top10_inc,
                        "avg_cons_cv_bot50_inc": avg_cons_cv_bot50_inc
                    })

                    # Single log call. use step=now_step to pin the wandb step to the epoch frame.
                    wandb.log(log_dict)

                    '''
                    percentiles = self.envs.stacked_data(self.eval_env.households.at_next.flatten())

                    # map indices -> decile ranges
                    decile_ranges = [
                        (90, 100),  # richest 10%
                        (80, 90),
                        (70, 80),
                        (60, 70),
                        (50, 60),
                        (40, 50),
                        (30, 40),
                        (20, 30),
                        (10, 20),
                        (0, 10)     # poorest 10%
                    ]

                    # Log each decile with unambiguous names
                    for (lo, hi), value in zip(decile_ranges, percentiles):
                        wandb.log({f"wealth_decile_{lo}_{hi}": float(value)})

                    wandb.log({"mean_households_utility": mean_house_rewards,
                               "goverment_utility": mean_gov_rewards,
                               "years": years,
                               "wealth_gini": avg_wealth_gini,
                               "income_gini": avg_income_gini,
                               "per_household_gdp": avg_gdp,
                               "tax_per_household": avg_mean_tax,
                               "tax_revenue_pct_gdp": tax_revenue_pct_gdp,
                               "post_income_per_household": avg_mean_post_income,
                               "wealth_per_household": avg_mean_wealth,
                               "government_actor_loss": gov_actor_loss,
                               "government_critic_loss": gov_critic_loss,
                               "households_actor_loss": house_actor_loss,
                               "households_critic_loss": house_critic_loss,
                               "ubi_prop_gdp": avg_ubi_prop_gdp,
                               "tax_minus_ubi_pct_gdp": tax_minus_ubi_pct_gdp,
                               "ubi_share_of_wealth": ubi_share_of_wealth,
                               "consumption_standard_deviation_overall": avg_cons_std_overall,
                               "consumption_coeff_variation_overall": avg_cons_cv_overall,
                               "consumption_standard_deviation_top10": avg_cons_std_top10,
                               "consumption_coeff_variation_top10": avg_cons_cv_top10,
                               "consumption_standard_deviation_bot50": avg_cons_std_bot50,
                               "consumption_coeff_variation_bot50": avg_cons_cv_bot50,
                               "pearson_corr_consumption_lag1_overall": avg_cons_lag1_overall,
                               "pearson_corr_consumption_lag1_top10": avg_cons_lag1_top10,
                               "pearson_corr_consumption_lag1_bot50": avg_cons_lag1_bot50,
                               "nominal_debt": avg_debt,
                               "debt_to_gdp_ratio": avg_debt_to_gdp,
                               "mean_working_hours": avg_mean_work_hours,
                               "labour_participation_rate": avg_participation,
                               "top10_mean_working_hours": avg_mean_work_top10,
                               "bottom50_mean_working_hours": avg_mean_work_bot50,
                               "mean_saving_propensity_overall": avg_mean_saving_prop,
                               "mean_saving_propensity_top10": avg_mean_saving_top10,
                               "mean_saving_propensity_bot50": avg_mean_saving_bot50,

                               })
                            '''
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, years:{:.3f}, gov_actor_loss: {:.3f}, gov_critic_loss: {:.3f}, house_actor_loss: {:.3f}, house_critic_loss: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_gov_rewards, mean_house_rewards,years, gov_actor_loss, gov_critic_loss, house_actor_loss, house_critic_loss))
                
                if mean_house_rewards > max_sw:
                    max_sw = mean_house_rewards
                    # save models
                    torch.save(self.gov_actor.state_dict(), str(self.model_path) + '/gov_actor.pt')
                    torch.save(self.house_actor.state_dict(), str(self.model_path) + '/house_actor.pt')
                    # self.save_true = True
                    self._evaluate_agent()
        
        self._flush_wealth_buffer()
        self._flush_income_pre_buffer()
        self._flush_income_post_buffer()

        if self.wandb:
            wandb.finish()

    def test(self):
        # self.gov_actor.load_state_dict(torch.load("/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/run105/gov_actor.pt"))
        # self.house_actor.load_state_dict(torch.load("/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/run105/house_actor.pt"))
        self.house_actor.load_state_dict(torch.load("/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/100/run23/house_actor.pt"))
        mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
        print("mean gov reward:", mean_gov_rewards)


    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        # past_mean_house_action = self.multiple_households_mean_action()
        for _ in range(self.args.init_exploration_steps):
            if exploration_policy == 'uniform':
                raise NotImplementedError
            elif exploration_policy == 'gaussian':
                with torch.no_grad():
                    gov_action = np.array([0.1/0.5, 0.0/0.05, 0, 0, 0.1, 0.8]) + np.random.normal(0,0.5, size=(6,))
                    temp = np.zeros((self.args.n_households, 2))
                    temp[:, 0] = 0.7
                    temp[:, 1] = 1 / 3
                    temp += np.random.normal(0,0.1, size=(self.args.n_households,2))

                    hou_action = temp * 2 - 1
                    gov_action = gov_action * 2 - 1
                    past_mean_house_action = self.multiple_households_mean_action(hou_action)[0]

                    action = {self.envs.government.name: gov_action,
                              self.envs.households.name: hou_action}
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                # store the episodes
                self.buffer.mf_add(global_obs, private_obs, gov_action, hou_action, past_mean_house_action,
                                gov_reward, house_reward, next_global_obs, next_private_obs, float(done))
                # past_mean_house_action = self.multiple_households_mean_action(hou_action)
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()
                    global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
                    # past_mean_house_action = self.multiple_households_mean_action()
        print("Initial exploration has been finished!")

    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    def _update_network(self, update_agent="households"):
        # smaple batch of samples from the replay buffer
        global_obses, private_obses, gov_actions, hou_actions, past_mean_house_actions, gov_rewards,\
        house_rewards, next_global_obses, next_private_obses, dones = self.buffer.mf_sample(self.args.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        hou_actions = torch.tensor(hou_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        past_mean_house_actions = torch.tensor(past_mean_house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        # update government critic
        next_gov_pi = self.gov_actor(next_global_obses)
        next_gov_action, _ = get_action_info(next_gov_pi, cuda=self.args.cuda).select_actions(reparameterize=True)

        # house_pis = self.house_actor(global_obses, private_obses, gov_actions, past_mean_house_actions, update=True)  # current action
        house_pis = self.house_actor(global_obses, private_obses, gov_actions, update=True)  # current action
        house_actions_info = get_action_info(house_pis, cuda=self.args.cuda)
        house_actions_, house_pre_tanh_value = house_actions_info.select_actions(reparameterize=True)

        # next_hou_pi = self.house_actor(next_global_obses, next_private_obses, next_gov_action, self.get_tensor_mean_action(house_actions_, private_obses), update=True)
        next_hou_pi = self.house_actor(next_global_obses, next_private_obses, next_gov_action, update=True)
        next_hou_action, _ = get_action_info(next_hou_pi, cuda=self.args.cuda).select_actions(reparameterize=True)

        gov_td_target = gov_rewards.reshape(self.args.batch_size,-1) + inverse_dones * self.args.gamma * self.target_gov_qf(next_global_obses, next_gov_action, self.get_tensor_mean_action(next_hou_action, next_private_obses))

        gov_q_value = self.gov_critic(global_obses, gov_actions, past_mean_house_actions)
        gov_td_delta = gov_td_target - gov_q_value
        gov_critic_loss = torch.mean(F.mse_loss(gov_q_value, gov_td_target.detach()))

        n_inverse_dones = inverse_dones.unsqueeze(1).repeat(1, self.args.n_households, 1)
        house_td_target = house_rewards + n_inverse_dones * self.args.gamma * self.target_house_qf(next_global_obses, next_private_obses, next_gov_action, next_hou_action, self.get_tensor_mean_action(next_hou_action, next_private_obses))
        house_q_value = self.house_critic(global_obses, private_obses, gov_actions, hou_actions, past_mean_house_actions)
        house_td_delta = house_td_target - house_q_value
        house_critic_loss = torch.mean(F.mse_loss(house_q_value, house_td_target.detach()))

        # government actor
        gov_pis = self.gov_actor(global_obses)
        gov_actions_info = get_action_info(gov_pis, cuda=self.args.cuda)
        gov_actions_, gov_pre_tanh_value = gov_actions_info.select_actions(reparameterize=True)
        gov_log_prob = gov_actions_info.get_log_prob(gov_actions_, gov_pre_tanh_value)
        gov_actor_loss = torch.mean(-gov_log_prob * gov_td_delta.detach())

        # households actor
        house_log_prob = house_actions_info.get_log_prob(house_actions_, house_pre_tanh_value)/self.args.n_households
        house_actor_loss = torch.mean(-house_log_prob.sum(2) * house_td_delta.detach().mean(1))


        if update_agent=="households":
            self.house_actor_optim.zero_grad()
            self.house_critc_optim.zero_grad()
            house_actor_loss.backward()
            house_critic_loss.backward()
            self.house_actor_optim.step()
            self.house_critc_optim.step()
        elif update_agent=="government":
            self.gov_actor_optim.zero_grad()
            self.gov_critic_optim.zero_grad()
            gov_actor_loss.backward()
            gov_critic_loss.backward()
            self.gov_actor_optim.step()
            self.gov_critic_optim.step()
        else: # update all
            self.house_actor_optim.zero_grad()
            self.house_critc_optim.zero_grad()
            house_actor_loss.backward()
            house_critic_loss.backward()
            self.house_actor_optim.step()
            self.house_critc_optim.step()

            self.gov_actor_optim.zero_grad()
            self.gov_critic_optim.zero_grad()
            gov_actor_loss.backward()
            gov_critic_loss.backward()
            self.gov_actor_optim.step()
            self.gov_critic_optim.step()

        return gov_actor_loss.item(), gov_critic_loss.item(), house_actor_loss.item(), house_critic_loss.item()

    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def _evaluate_agent(self):
        total_gov_reward = 0
        total_house_reward = 0
        total_steps = 0
        mean_tax = 0
        mean_wealth = 0
        mean_post_income = 0
        gdp = 0
        income_gini = 0
        wealth_gini = 0
        ubi_prop_gdp = 0.0

        cons_std_overall = 0.0
        cons_cv_overall = 0.0
        cons_std_top10 = 0.0
        cons_cv_top10 = 0.0
        cons_std_bot50 = 0.0
        cons_cv_bot50 = 0.0

        cons_lag1_overall = 0.0
        cons_lag1_top10 = 0.0
        cons_lag1_bot50 = 0.0

        mean_debt = 0.0
        mean_debt_to_gdp = 0.0

        mean_work_hours = 0.0
        mean_participation = 0.0
        mean_work_top10 = 0.0
        mean_work_bot50 = 0.0

        mean_saving_prop = 0.0
        mean_saving_top10 = 0.0
        mean_saving_bot50 = 0.0
        penalty_freq = 0.0

        cons_cv_overall_inc = 0.0
        cons_cv_top10_inc = 0.0
        cons_cv_bot50_inc = 0.0

        tax_gdp_ratio = 0.0

        tax_share_top10 = 0.0
        tax_share_bot50 = 0.0

        tax_burden_overall = 0.0
        tax_burden_top10 = 0.0
        tax_burden_bot50 = 0.0

        income_tax_rate_overall = 0.0
        income_tax_rate_top10 = 0.0
        income_tax_rate_bot50 = 0.0

        wealth_tax_rate_overall = 0.0
        wealth_tax_rate_top10 = 0.0
        wealth_tax_rate_bot50 = 0.0

        episode_tax_share_top10 = 0.0
        episode_tax_share_bot50 = 0.0
        episode_tax_burden_overall = 0.0
        episode_tax_burden_top10 = 0.0
        episode_tax_burden_bot50 = 0.0

        ep_inc_rate_overall = ep_inc_rate_top10 = ep_inc_rate_bot50 = 0.0
        ep_wealth_rate_overall = ep_wealth_rate_top10 = ep_wealth_rate_bot50 = 0.0



        for epoch_i in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
            episode_gov_reward = 0
            episode_mean_house_reward = 0
            step_count = 0
            episode_mean_tax = []
            episode_mean_wealth = []
            episode_mean_post_income = []
            episode_gdp = []
            episode_income_gini = []
            episode_wealth_gini = []
            episode_ubi_prop_gdp = []
            episode_consumption = []

            episode_debt = []
            episode_debt_to_gdp = []

            episode_mean_work_hours = []
            episode_participation = []
            episode_work_top10 = []
            episode_work_bot50 = []

            episode_hts = []
            episode_saving_p = []
            episode_tax_gdp_ratio = []


            penalty_years = 0
            total_years = 0

            while True:
                with torch.no_grad():
                    action = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                step_count += 1
                episode_gov_reward += gov_reward
                episode_mean_house_reward += np.mean(house_reward)
                episode_mean_tax.append(np.mean(self.eval_env.tax_array))
                episode_mean_wealth.append(np.mean(self.eval_env.households.at_next))
                episode_mean_post_income.append(np.mean(self.eval_env.post_income))
                episode_gdp.append(self.eval_env.per_household_gdp)
                episode_income_gini.append(self.eval_env.income_gini)
                episode_wealth_gini.append(self.eval_env.wealth_gini)

                tax_t = float(np.mean(self.eval_env.tax_array))
                gdp_t = float(self.eval_env.per_household_gdp)

                episode_tax_gdp_ratio.append(tax_t / max(1e-8, gdp_t))

                # ================================
                # Distributional tax metrics
                # ================================
                tax = self.eval_env.tax_array.flatten()
                income = self.eval_env.income.flatten()
                wealth = self.eval_env.households.at.flatten()

                # guard
                total_tax = np.sum(tax)
                total_income = np.sum(income)

                if total_tax > 0 and total_income > 0:
                    # group indices (wealth-based, consistent with env)
                    sorted_idx = np.argsort(wealth)[::-1]
                    top10_idx = sorted_idx[:max(1, int(0.1 * len(sorted_idx)))]
                    bot50_idx = sorted_idx[int(0.5 * len(sorted_idx)):]

                    # ---- Metric A: Tax share
                    episode_tax_share_top10 = np.sum(tax[top10_idx]) / total_tax
                    episode_tax_share_bot50 = np.sum(tax[bot50_idx]) / total_tax

                    # ---- Metric B: Effective tax rate
                    resources = income + wealth

                    tax_burden = np.where(resources > 0, tax / resources, 0.0)

                    episode_tax_burden_overall = np.mean(tax_burden)
                    episode_tax_burden_top10 = np.mean(tax_burden[top10_idx])
                    episode_tax_burden_bot50 = np.mean(tax_burden[bot50_idx])

                else:
                    episode_tax_share_top10 = episode_tax_share_bot50 = 0.0
                    episode_tax_burden_overall = episode_tax_burden_top10 = episode_tax_burden_bot50 = 0.0

                income_tax = self.eval_env.income_tax.flatten()
                wealth_tax = self.eval_env.asset_tax.flatten()

                total_income = np.sum(income)
                total_wealth = np.sum(wealth)

                if total_income > 0:
                    ep_inc_rate_overall = np.sum(income_tax) / total_income
                    ep_inc_rate_top10 = np.sum(income_tax[top10_idx]) / max(1e-8, np.sum(income[top10_idx]))
                    ep_inc_rate_bot50 = np.sum(income_tax[bot50_idx]) / max(1e-8, np.sum(income[bot50_idx]))
                else:
                    ep_inc_rate_overall = ep_inc_rate_top10 = ep_inc_rate_bot50 = 0.0

                if total_wealth > 0:
                    ep_wealth_rate_overall = np.sum(wealth_tax) / total_wealth
                    ep_wealth_rate_top10 = np.sum(wealth_tax[top10_idx]) / max(1e-8, np.sum(wealth[top10_idx]))
                    ep_wealth_rate_bot50 = np.sum(wealth_tax[bot50_idx]) / max(1e-8, np.sum(wealth[bot50_idx]))
                else:
                    ep_wealth_rate_overall = ep_wealth_rate_top10 = ep_wealth_rate_bot50 = 0.0


                # --- working hours metrics (collect per-step)
                ht = self.eval_env.ht.flatten()                      # actual hours (shape (N,))
                mean_ht = float(np.mean(ht))
                participation = float(np.mean(ht > (1e-8)))

                # collect per-step summaries and full vector for later grouping
                episode_mean_work_hours.append(mean_ht)
                episode_participation.append(participation)
                episode_hts.append(ht)    # collect full vector per timestep
                episode_saving_p.append(self.eval_env.saving_p.flatten())


                _ubi_prop_gdp = float(getattr(self.eval_env, "ubi_prop_gdp", 0.0))

                episode_ubi_prop_gdp.append(float(_ubi_prop_gdp))

                # record current period debt (use Bt_next computed by env.step) and debt-to-GDP
                episode_debt.append(float(getattr(self.eval_env, "Bt_next", self.eval_env.Bt)))
                # guard against zero GDP
                episode_debt_to_gdp.append(float(getattr(self.eval_env, "Bt_next", self.eval_env.Bt)) /
                                           max(1e-8, float(getattr(self.eval_env, "GDP", self.eval_env.per_household_gdp))))

                episode_consumption.append(self.eval_env.consumption.flatten())
                if self.result_evaluation == True:
                    if step_count == 1 or step_count == 100 or step_count == 200 or step_count == 300:
                        save_parameters(self.model_path, step_count, epoch_i, self.eval_env)

                total_years += 1
                if getattr(self.eval_env, "excess_penalty_applied", False):
                    penalty_years += 1
                if done:
                    break

                global_obs = next_global_obs
                private_obs = next_private_obs

            total_gov_reward += episode_gov_reward
            total_house_reward += episode_mean_house_reward
            total_steps += step_count
            mean_tax += np.mean(episode_mean_tax)
            mean_wealth += np.mean(episode_mean_wealth)
            mean_post_income += np.mean(episode_mean_post_income)
            gdp += np.mean(episode_gdp)
            income_gini += np.mean(episode_income_gini)
            wealth_gini += np.mean(episode_wealth_gini)

            ubi_prop_gdp += np.mean(episode_ubi_prop_gdp)
            tax_gdp_ratio += np.mean(episode_tax_gdp_ratio)

            tax_share_top10 += episode_tax_share_top10
            tax_share_bot50 += episode_tax_share_bot50

            tax_burden_overall += episode_tax_burden_overall
            tax_burden_top10 += episode_tax_burden_top10
            tax_burden_bot50 += episode_tax_burden_bot50

            income_tax_rate_overall += ep_inc_rate_overall
            income_tax_rate_top10 += ep_inc_rate_top10
            income_tax_rate_bot50 += ep_inc_rate_bot50

            wealth_tax_rate_overall += ep_wealth_rate_overall
            wealth_tax_rate_top10 += ep_wealth_rate_top10
            wealth_tax_rate_bot50 += ep_wealth_rate_bot50





            if total_years > 0:
                penalty_freq += penalty_years / total_years

            if len(episode_debt) > 0:
                mean_debt += np.mean(episode_debt)
                mean_debt_to_gdp += np.mean(episode_debt_to_gdp)
            else:
                mean_debt += 0.0
                mean_debt_to_gdp += 0.0


            if len(episode_consumption) > 0:
                cons_mat = np.vstack(episode_consumption)                     # shape (T, N)
                per_house_std = np.std(cons_mat, axis=0)                      # volatility per household
                per_house_mean = np.mean(cons_mat, axis=0)
                # avoid division by zero for CV
                per_house_cv = np.where(per_house_mean > 0, per_house_std / per_house_mean, 0.0)

                # group indices by final wealth (consistent with env grouping logic)
                '''
                wealth = self.eval_env.households.at_next.flatten()
                sorted_idx = np.argsort(wealth)[::-1]                         # descending wealth
                top10_idx = sorted_idx[:max(1, int(0.1 * len(sorted_idx)))]
                bot50_idx = sorted_idx[int(0.5 * len(sorted_idx)):]

                                # ===== Option B: income-conditional consumption volatility =====
                # use post-tax income (what households actually consume out of)
                income = self.eval_env.post_income.flatten()
                '''
                sorted_inc_idx = np.argsort(income)[::-1]
                top10_inc_idx = sorted_inc_idx[:max(1, int(0.1 * len(sorted_inc_idx)))]
                bot50_inc_idx = sorted_inc_idx[int(0.5 * len(sorted_inc_idx)):]

                episode_cons_cv_top10_inc = np.mean(per_house_cv[top10_inc_idx])
                episode_cons_cv_bot50_inc = np.mean(per_house_cv[bot50_inc_idx])
                episode_cons_cv_overall_inc = np.mean(per_house_cv)

                # episode-level group stats: mean of per-household volatilities (std and CV)
                episode_cons_std_overall = np.mean(per_house_std)
                episode_cons_cv_overall = np.mean(per_house_cv)

                episode_cons_std_top10 = np.mean(per_house_std[top10_idx])
                episode_cons_cv_top10 = np.mean(per_house_cv[top10_idx])

                episode_cons_std_bot50 = np.mean(per_house_std[bot50_idx])
                episode_cons_cv_bot50 = np.mean(per_house_cv[bot50_idx])

                if cons_mat.shape[0] > 1:
                    cons_t0 = cons_mat[:-1, :]   # time 0..T-2
                    cons_t1 = cons_mat[1:, :]    # time 1..T-1

                    def _safe_corr(x, y):
                        xm = x - x.mean()
                        ym = y - y.mean()
                        denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
                        return np.sum(xm * ym) / denom if denom > 0 else 0.0

                    per_house_lag1 = np.array([_safe_corr(cons_t0[:, i], cons_t1[:, i]) for i in range(cons_mat.shape[1])])
                else:
                    per_house_lag1 = np.zeros(cons_mat.shape[1])

                episode_lag1_overall = np.mean(per_house_lag1)
                episode_lag1_top10 = np.mean(per_house_lag1[top10_idx])
                episode_lag1_bot50 = np.mean(per_house_lag1[bot50_idx])

            else:
                episode_cons_std_overall = episode_cons_cv_overall = 0.0
                episode_cons_std_top10 = episode_cons_cv_top10 = 0.0
                episode_cons_std_bot50 = episode_cons_cv_bot50 = 0.0
            
            if len(episode_hts) > 0:
                ht_mat = np.vstack(episode_hts)             # shape (T, N)
                per_house_mean_ht = np.mean(ht_mat, axis=0) # mean hours per household across episode

                '''# group indices by final wealth (consistent with consumption grouping)
                wealth = self.eval_env.households.at_next.flatten()
                sorted_idx = np.argsort(wealth)[::-1]                         # descending wealth
                top10_idx = sorted_idx[:max(1, int(0.1 * len(sorted_idx)))]
                bot50_idx = sorted_idx[int(0.5 * len(sorted_idx)):]
                '''

                # episode-level grouped stats
                episode_work_top10 = np.mean(per_house_mean_ht[top10_idx])
                episode_work_bot50 = np.mean(per_house_mean_ht[bot50_idx])
                episode_consistent_mean_work = np.mean(per_house_mean_ht)    # alternative to averaging timestep means
            else:
                episode_work_top10 = episode_work_bot50 = episode_consistent_mean_work = 0.0

            if len(episode_saving_p) > 0:
                save_mat = np.vstack(episode_saving_p)              # shape (T, N)
                per_house_mean_save = np.mean(save_mat, axis=0)     # mean saving propensity per household across episode

                '''# group indices by final wealth (same logic as consumption)
                wealth = self.eval_env.households.at_next.flatten()
                sorted_idx = np.argsort(wealth)[::-1]                         # descending wealth
                top10_idx = sorted_idx[:max(1, int(0.1 * len(sorted_idx)))]
                bot50_idx = sorted_idx[int(0.5 * len(sorted_idx)):]'''

                episode_save_overall = np.mean(per_house_mean_save)
                episode_save_top10 = np.mean(per_house_mean_save[top10_idx])
                episode_save_bot50 = np.mean(per_house_mean_save[bot50_idx])
            else:
                episode_save_overall = episode_save_top10 = episode_save_bot50 = 0.0

            # choose which episode mean to accumulate:
            # ep_mean_work = np.mean(episode_mean_work_hours)     # your current approach (avg of per-step means)
            ep_mean_work = episode_consistent_mean_work            # recommended: mean of per-household average hours
            ep_participation = np.mean(episode_participation)
            ep_work_top10 = episode_work_top10
            ep_work_bot50 = episode_work_bot50


            cons_std_overall += episode_cons_std_overall
            cons_cv_overall  += episode_cons_cv_overall
            cons_std_top10   += episode_cons_std_top10
            cons_cv_top10    += episode_cons_cv_top10
            cons_std_bot50   += episode_cons_std_bot50
            cons_cv_bot50    += episode_cons_cv_bot50

            cons_cv_overall_inc += episode_cons_cv_overall_inc
            cons_cv_top10_inc += episode_cons_cv_top10_inc
            cons_cv_bot50_inc += episode_cons_cv_bot50_inc

            cons_lag1_overall += episode_lag1_overall
            cons_lag1_top10 += episode_lag1_top10
            cons_lag1_bot50 += episode_lag1_bot50

            mean_work_hours += ep_mean_work
            mean_participation += ep_participation
            mean_work_top10 += ep_work_top10
            mean_work_bot50 += ep_work_bot50

            mean_saving_prop += episode_save_overall
            mean_saving_top10 += episode_save_top10
            mean_saving_bot50 += episode_save_bot50

        avg_gov_reward = total_gov_reward / self.args.eval_episodes
        avg_house_reward = total_house_reward / self.args.eval_episodes
        mean_step = total_steps / self.args.eval_episodes
        avg_mean_tax = mean_tax / self.args.eval_episodes
        avg_mean_wealth = mean_wealth / self.args.eval_episodes
        avg_mean_post_income = mean_post_income / self.args.eval_episodes
        avg_gdp = gdp / self.args.eval_episodes
        avg_income_gini = income_gini / self.args.eval_episodes
        avg_wealth_gini = wealth_gini / self.args.eval_episodes

        avg_ubi_prop_gdp = ubi_prop_gdp / self.args.eval_episodes
        avg_cons_std_overall = cons_std_overall / self.args.eval_episodes
        avg_cons_cv_overall  = cons_cv_overall  / self.args.eval_episodes
        avg_cons_std_top10   = cons_std_top10   / self.args.eval_episodes
        avg_cons_cv_top10    = cons_cv_top10    / self.args.eval_episodes
        avg_cons_std_bot50   = cons_std_bot50   / self.args.eval_episodes
        avg_cons_cv_bot50    = cons_cv_bot50    / self.args.eval_episodes

        avg_cons_lag1_overall = cons_lag1_overall / self.args.eval_episodes
        avg_cons_lag1_top10 = cons_lag1_top10 / self.args.eval_episodes
        avg_cons_lag1_bot50 = cons_lag1_bot50 / self.args.eval_episodes

        avg_debt = (mean_debt / self.args.eval_episodes) *(-1)
        avg_debt_to_gdp = (mean_debt_to_gdp / self.args.eval_episodes) *(-1)

        avg_mean_work_hours = mean_work_hours / self.args.eval_episodes
        avg_participation = mean_participation / self.args.eval_episodes
        avg_mean_work_top10 = mean_work_top10 / self.args.eval_episodes
        avg_mean_work_bot50 = mean_work_bot50 / self.args.eval_episodes

        avg_mean_saving_prop = mean_saving_prop / self.args.eval_episodes
        avg_mean_saving_top10 = mean_saving_top10 / self.args.eval_episodes
        avg_mean_saving_bot50 = mean_saving_bot50 / self.args.eval_episodes

        avg_cons_cv_overall_inc = cons_cv_overall_inc / self.args.eval_episodes
        avg_cons_cv_top10_inc = cons_cv_top10_inc / self.args.eval_episodes
        avg_cons_cv_bot50_inc = cons_cv_bot50_inc / self.args.eval_episodes

        avg_penalty_freq = penalty_freq / self.args.eval_episodes
        avg_tax_gdp_ratio = tax_gdp_ratio / self.args.eval_episodes

        avg_tax_share_top10 = tax_share_top10 / self.args.eval_episodes
        avg_tax_share_bot50 = tax_share_bot50 / self.args.eval_episodes

        avg_tax_burden_overall = tax_burden_overall / self.args.eval_episodes
        avg_tax_burden_top10 = tax_burden_top10 / self.args.eval_episodes
        avg_tax_burden_bot50 = tax_burden_bot50 / self.args.eval_episodes

        avg_income_tax_rate_overall = income_tax_rate_overall / self.args.eval_episodes
        avg_income_tax_rate_top10 = income_tax_rate_top10 / self.args.eval_episodes
        avg_income_tax_rate_bot50 = income_tax_rate_bot50 / self.args.eval_episodes

        avg_wealth_tax_rate_overall = wealth_tax_rate_overall / self.args.eval_episodes
        avg_wealth_tax_rate_top10 = wealth_tax_rate_top10 / self.args.eval_episodes
        avg_wealth_tax_rate_bot50 = wealth_tax_rate_bot50 / self.args.eval_episodes






        return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, \
               avg_wealth_gini, mean_step, avg_ubi_prop_gdp, avg_cons_std_overall, avg_cons_cv_overall, avg_cons_std_top10, avg_cons_cv_top10, \
                avg_cons_std_bot50, avg_cons_cv_bot50, avg_cons_lag1_overall, avg_cons_lag1_top10, avg_cons_lag1_bot50, avg_debt, avg_debt_to_gdp, \
                avg_mean_work_hours, avg_participation, avg_mean_work_top10, avg_mean_work_bot50, avg_mean_saving_prop, avg_mean_saving_top10, avg_mean_saving_bot50, \
                    avg_penalty_freq, avg_cons_cv_overall_inc, avg_cons_cv_top10_inc, avg_cons_cv_bot50_inc, avg_tax_gdp_ratio, avg_tax_share_top10, avg_tax_share_bot50, \
                    avg_tax_burden_overall, avg_tax_burden_top10, avg_tax_burden_bot50, avg_income_tax_rate_overall, avg_income_tax_rate_top10, avg_income_tax_rate_bot50, avg_wealth_tax_rate_overall, avg_wealth_tax_rate_top10, avg_wealth_tax_rate_bot50


    def _evaluate_get_action(self, global_obs, private_obs):
        global_obs_tensor = self._get_tensor_inputs(global_obs)
        private_obs_tensor = self._get_tensor_inputs(private_obs)
        gov_pi = self.gov_actor(global_obs_tensor)
        gov_action = get_action_info(gov_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
        hou_pi = self.house_actor(global_obs_tensor, private_obs_tensor, gov_action)
        hou_action = get_action_info(hou_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
        gov_action = gov_action.cpu().numpy()[0]
        hou_action = hou_action.cpu().numpy()[0]

        action = {self.envs.government.name: gov_action,
                  self.envs.households.name: hou_action}
        return action

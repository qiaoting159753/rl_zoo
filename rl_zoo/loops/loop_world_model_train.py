import torch
import logging
import numpy as np
from tqdm import trange
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm
import math
from rl_zoo.utils import normalize

logging.basicConfig(level=logging.INFO)

# Agents
from rl_zoo.networks.mfrl.common import Actor
from rl_zoo.networks.mfrl.sac import SAC_Critic
from rl_zoo.agents.mfrl import SAC
from rl_zoo.agents.mbrl import Dyna_SAC_NS
from rl_zoo.utils import PrioritizedReplayBuffer

# World Models
from rl_zoo.networks.world_models.deterministic import (Single_PNN,
                                                        Prior_World_Model)

from rl_zoo.networks.world_models.ensembles import (Ensemble_Dyna_Ensemble_Reward,
                                                    Ensemble_Dyna_One_Reward)

from rl_zoo.networks.world_models.bayesian import (Bayesian_World_Model_BBB,
                                                   Bayesian_World_Model_LA,
                                                   Bayesian_World_Model_SGLD)


class World_Model_Trainer:
    """
    Training and evaluation loop for Model-Based agents that does not need to train the world model.
    """

    def __init__(self,
                 env,
                 world_model_name: str,
                 on_policy: bool,
                 random_goal: bool,
                 device: str,
                 G: int,
                 model_G: float,
                 batch_size: int,
                 episode_steps: int,
                 maximum_steps: int,
                 evaluate_interval: int,
                 generate_results: bool,
                 seed: int,
                 directory: str,
                 sas: bool,
                 prob_rwd: bool,
                 train_both: bool,
                 train_reward: bool,
                 parameter_a: float,
                 parameter_b: float,
                 parameter_c: float):
        # Training
        self.counter = 0
        self.G = G
        self.model_G = model_G
        self.batch_size = batch_size
        self.maximum_steps = maximum_steps
        self.episode_steps = episode_steps
        self.parameter_a = parameter_a
        self.parameter_b = parameter_b
        self.parameter_c = parameter_c
        # Environment
        self.env = env
        self.state_dim = env.observation_space
        self.action_dim = env.action_num
        self.random_goal = random_goal
        self.sas = sas
        self.prob_rwd = prob_rwd
        self.train_both = train_both
        self.train_reward = train_reward
        # Save data
        self.evaluate_interval = evaluate_interval
        self.evaluation_array = []
        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        self.directory = directory
        self.seed = seed
        self.generate_results = generate_results

        # Agent and world model.
        self.world_model_name = world_model_name
        self.agent = None
        self.world_model = None
        self.on_policy = on_policy
        self.memory = PrioritizedReplayBuffer()
        self.device = device
        self.agent_selection()

        logging.info(f"Name: {self.world_model_name}, On Policy: {on_policy}, Random Goal:{random_goal}")

    def evaluate(self):
        """
        Evaluation
        """
        # Evaluate 10 times
        l2_one_step_errors = []
        l1_one_step_errors = []
        l1_one_rwd_errors = []
        l1_one_gt_rwd_errors = []
        one_dyna_uncerts = []
        one_rwd_uncerts = []

        gt_s = self.env.reset()
        episodic_pred_error = 0.0
        episodic_rwd_pred_error = 0.0

        for _ in range(self.episode_steps):
            if self.on_policy:
                action = self.agent.select_action_from_policy(gt_s)
            else:
                action = self.env.sample_action()

            action = normalize(action, self.env.max_action_value, self.env.min_action_value)

            gt_ns, gt_rwd, gt_done, _ = self.env.step(action)
            # Converting to tensor
            tensor_action = torch.FloatTensor(action).to(self.device).unsqueeze(dim=0)
            tensor_state = torch.FloatTensor(gt_s).to(self.device).unsqueeze(dim=0)
            # One step prediction
            pred_ns, _, _, _ = self.world_model.pred_next_states(observation=tensor_state,
                                                                 actions=tensor_action)
            # MSE. L1 of dynamics
            np_pred_ns = pred_ns.detach().squeeze().cpu().numpy()
            one_step_mse = (np.square(np_pred_ns - gt_ns)).mean()
            one_step_l1 = (abs(np_pred_ns - gt_ns)).mean()
            l1_one_step_errors.append(one_step_l1)
            l2_one_step_errors.append(one_step_mse)
            episodic_pred_error += one_step_mse

            if self.train_reward:
                pred_reward, _ = self.world_model.pred_rewards(observation=tensor_state,
                                                               action=tensor_action,
                                                               next_observation=pred_ns)
                pred_reward = pred_reward.detach().squeeze().cpu().numpy()
            else:
                # Ground Truth Reward function prediction
                pred_reward = 0.0
            if self.train_reward:
                l1_one_rwd_error = abs(pred_reward - gt_rwd)
                l1_one_rwd_errors.append(l1_one_rwd_error)
                episodic_rwd_pred_error += l1_one_rwd_error
            else:
                l1_one_rwd_errors.append(0.0)

            pred_gt_reward = self.env.get_gt_reward(gt_s.squeeze(), action.squeeze(),
                                                    pred_ns.detach().cpu().numpy().squeeze())
            l1_one_gt_rwd_error = abs(pred_gt_reward - gt_rwd)
            l1_one_gt_rwd_errors.append(l1_one_gt_rwd_error)

            # Errors are done: L1 WM, L2 WM, L1 PRED RW, L1 GT RW.
            one_dyna_uncert, one_rwd_uncert, samples = self.world_model.estimate_uncertainty(observation=tensor_state,
                                                                                             actions=tensor_action,
                                                                                             train_reward=self.train_reward)
            one_dyna_uncerts.append(one_dyna_uncert)
            # Reward uncertainty estimation
            if samples is None:
                one_rwd_uncerts.append(one_rwd_uncert)
            else:
                samples = samples.detach().cpu().numpy()
                rds = []
                for i in range(samples.shape[0]):
                    sample = samples[i]
                    pred_s_reward = self.env.get_gt_reward(gt_s.squeeze(), action.squeeze(), sample)
                    rds.append(pred_s_reward)
                rds = np.array(rds)
                one_rwd_uncert = np.var(rds, axis=0)
                one_rwd_uncerts.append(one_rwd_uncert)
            #################    Uncertainty Estimation and Quantification    ################
            gt_s = gt_ns
            if gt_done:
                break
        l1_one_step_errors = np.array(l1_one_step_errors)
        l2_one_step_errors = np.array(l2_one_step_errors)
        l1_one_rwd_errors = np.array(l1_one_rwd_errors)
        l1_one_gt_rwd_errors = np.array(l1_one_gt_rwd_errors)
        one_dyna_uncerts = np.array(one_dyna_uncerts)
        one_rwd_uncerts = np.array(one_rwd_uncerts)

        c_1 = np.corrcoef(l2_one_step_errors, one_dyna_uncerts)
        c_2 = np.corrcoef(l1_one_step_errors, one_dyna_uncerts)
        c_3 = np.corrcoef(l1_one_rwd_errors, one_rwd_uncerts)
        c_4 = np.corrcoef(l1_one_gt_rwd_errors, one_rwd_uncerts)

        logging.info(f"Prediction Error dynamics: {episodic_pred_error}, reward:{episodic_rwd_pred_error}")
        all_data = np.zeros((7,))
        all_data[0] = self.counter
        all_data[1] = episodic_pred_error
        all_data[2] = episodic_rwd_pred_error
        if math.isnan(c_1[0, 1]):
            c_1[0, 1] = 0.0
        if math.isnan(c_2[0, 1]):
            c_2[0, 1] = 0.0
        if math.isnan(c_3[0, 1]):
            c_3[0, 1] = 0.0
        if math.isnan(c_4[0, 1]):
            c_4[0, 1] = 0.0
        all_data[3] = c_1[0, 1]
        all_data[4] = c_2[0, 1]
        all_data[5] = c_3[0, 1]
        all_data[6] = c_4[0, 1]

        self.evaluation_array.append(all_data)

        if self.generate_results:
            # Save the metrics
            file_name = self.directory + str(self.seed) + "_" + self.date_and_time + ".csv"
            np.savetxt(file_name + ".csv", np.array(self.evaluation_array), delimiter=",")

    def train(self, flush=False):
        """
        Train the MFRL Agent.
        """
        with logging_redirect_tqdm():
            need_evaluate = False
            need_reset = True
            for _ in trange(self.maximum_steps):
                if need_reset:
                    epi_reward = 0.0
                    step_counter = 0
                    state = self.env.reset()
                    need_reset = False
                if self.on_policy:
                    action = self.agent.select_action_from_policy(state)
                else:
                    action = self.env.sample_action()
                # Do action is for the environment.
                next_state, reward, done, _ = self.env.step(action)
                # Small action is for training.
                self.memory.add(state, action, reward, next_state, done)
                epi_reward += reward
                step_counter += 1
                state = next_state
                # Training
                if len(self.memory) > self.batch_size:
                    # first time update world model statistics
                    if len(self.memory) == (self.batch_size + 1):
                        statistics = self.memory.get_statistics()
                        self.world_model.set_statistics(statistics)
                    # Train the agent only when on-policy.
                    if self.on_policy:
                        for _ in range(self.G):
                            self.agent.train_policy(self.memory, batch_size=self.batch_size)
                    self.counter += 1
                    # Train the world model every time.
                    if self.model_G > 1.0:
                        for _ in range(int(self.model_G)):
                            self.agent.train_world_model(memory=self.memory,
                                                               batch_size=self.batch_size,
                                                               )
                    else:
                        # For every a few steps
                        if self.counter % (int(1.0 / self.model_G)) == 0:
                            self.agent.train_world_model(memory=self.memory,
                                                               batch_size=self.batch_size,
                                                               )
                # Evaluating
                if (self.counter % self.evaluate_interval == 0) and (self.counter > self.batch_size):
                    need_evaluate = True
                # End of Episode
                if done or ((step_counter % self.episode_steps) == 0):
                    # Reset at next
                    logging.info(f"Training:{epi_reward}")
                    need_reset = True
                    # Update World Model Statistics.
                    if len(self.memory) > self.batch_size:
                        statistics = self.memory.get_statistics()
                        self.world_model.set_statistics(statistics)
                    # Evaluation
                    if need_evaluate:
                        self.evaluate()
                        need_evaluate = False

    def agent_selection(self):
        """
        Create an agent
        """
        actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
        critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
        # self.agent = SAC(actor_network=actor,
        #                  critic_network=critic,
        #                  action_num=self.action_dim,
        #                  alpha_lr=3e-4,
        #                  gamma=0.99,
        #                  tau=0.005,
        #                  actor_lr=3e-4,
        #                  critic_lr=3e-4,
        #                  device=torch.device(self.device),
        #                  reward_scale=1.0)

        if self.world_model_name == "Single_PNN":
            self.world_model = Single_PNN(observation_size=self.state_dim,
                                          num_actions=self.action_dim,
                                          device=self.device,
                                          sas=self.sas,
                                          prob_rwd=self.prob_rwd
                                          )

        if self.world_model_name == "Prior_World_Model":
            self.world_model = Prior_World_Model(observation_size=self.state_dim,
                                                 num_actions=self.action_dim,
                                                 device=self.device,
                                                 sas=self.sas,
                                                 prob_rwd=self.prob_rwd
                                                 )

        if self.world_model_name == "Ensemble_Dyna_One_Reward":
            self.world_model = Ensemble_Dyna_One_Reward(observation_size=self.state_dim,
                                                        num_actions=self.action_dim,
                                                        device=self.device,
                                                        boost_inter=int(self.parameter_a),
                                                        sas=self.sas,
                                                        prob_rwd=self.prob_rwd)

        if self.world_model_name == "Ensemble_Dyna_Ensemble_Reward":
            self.world_model = Ensemble_Dyna_Ensemble_Reward(observation_size=self.state_dim,
                                                             num_actions=self.action_dim,
                                                             device=self.device,
                                                             boost_inter=int(self.parameter_a),
                                                             sas=self.sas,
                                                             prob_rwd=self.prob_rwd)

        if self.world_model_name == "Bayesian_VI":
            # Ratio: Small is better: 0.3, 0.1
            self.world_model = Bayesian_World_Model_BBB(observation_size=self.state_dim,
                                                        num_actions=self.action_dim,
                                                        device=self.device,
                                                        ratio=self.parameter_a,
                                                        sigma=self.parameter_b,
                                                        sas=self.sas,
                                                        prob_rwd=self.prob_rwd)

        if self.world_model_name == "Bayesian_Laplace":
            # no change on temp, sigma
            self.world_model = Bayesian_World_Model_LA(observation_size=self.state_dim,
                                                       num_actions=self.action_dim,
                                                       device=self.device,
                                                       temperature=self.parameter_a,
                                                       prior_precision=self.parameter_b,
                                                       sas=self.sas,
                                                       prob_rwd=self.prob_rwd)

        if self.world_model_name == "Bayesian_World_Model_SGLD":
            self.world_model = Bayesian_World_Model_SGLD(observation_size=self.state_dim,
                                                         num_actions=self.action_dim,
                                                         device=self.device,
                                                         sas=self.sas,
                                                         prob_rwd=self.prob_rwd
                                                         )

        self.agent = SAC(actor_network=actor,
                         critic_network=critic,
                         world_model= self.world_model,
                         action_num=self.action_dim,
                         alpha_lr=3e-4,
                         gamma=0.99,
                         tau=0.005,
                         actor_lr=3e-4,
                         critic_lr=3e-4,
                         device=torch.device(self.device),
                         reward_scale=1.0)

        # self.agent = Dyna_SAC_NS(actor_network=actor,
        #                          critic_network=critic,
        #                          world_network=self.world_model,
        #                          num_samples=10,
        #                          action_num=self.action_dim,
        #                          alpha_lr=3e-4,
        #                          horizon=1,
        #                          gamma=0.99,
        #                          tau=0.005,
        #                          actor_lr=3e-4,
        #                          critic_lr=3e-4,
        #                          device=torch.device(self.device))

        # if self.world_model_name == "Ensemble_Dyna_One_SAS_Reward":
        #     self.world_model = Ensemble_Dyna_One_SAS_Reward(observation_size=self.state_dim,
        #                                                     num_actions=self.action_dim,
        #                                                     num_models=5,
        #                                                     l_r=0.001,
        #                                                     device=self.device)
        # if self.world_model_name == "One_Dyna_One_SAS_Reward":
        #     self.world_model = One_Dyna_One_SAS_Reward(observation_size=self.state_dim,
        #                                                num_actions=self.action_dim,
        #                                                l_r=0.001,
        #                                                device=self.device)
        # if self.world_model_name == "Gaussian_Process":
        #     self.world_model = Gaussian_Process_World_Model(observation_size=self.state_dim,
        #                                                     num_actions=self.action_dim,
        #                                                     l_r=0.001, device=self.device, noise=self.parameter_a,
        #                                                     train_iter=int(self.parameter_b))
        # if self.world_model_name == "Bayesian_LR":
        #     self.world_model = Bayesian_World_Model_BBB(observation_size=self.state_dim, num_actions=self.action_dim,
        #                                                 l_r=0.001, device=self.device, option=2, sigma=self.parameter_a,
        #                                                 ratio=self.parameter_b)
        # if self.world_model_name == "Hyper_Bayesian_VI":
        #     self.world_model = Bayesian_World_Model_BBB(observation_size=self.state_dim,
        #                                                 num_actions=self.action_dim,
        #                                                 l_r=0.001,
        #                                                 device=self.device,
        #                                                 option=0,
        #                                                 sigma=self.parameter_a,
        #                                                 ratio=self.parameter_b)
        # if self.world_model_name == "Bayesian_Laplace_JA":
        #     self.world_model = Bayesian_World_Model_Laplace_JA(observation_size=self.state_dim,
        #                                                        num_actions=self.action_dim,
        #                                                        l_r=0.001,
        #                                                        hidden_size=128,
        #                                                        device=self.device)
        # if self.world_model_name == "Conditional_NF_NVP":
        #     self.world_model = Conditional_NVP_World_Model(observation_size=self.state_dim,
        #                                                    num_actions=self.action_dim,
        #                                                    l_r=0.001,
        #                                                    device=self.device)
        #
        # if self.world_model_name == "Ensemble_NF_NVP":
        #     self.world_model = Ensemble_NF_One_SAS_Reward(num_models=5,
        #                                                   observation_size=self.state_dim,
        #                                                   num_actions=self.action_dim,
        #                                                   l_r=0.00002,
        #                                                   device=self.device)

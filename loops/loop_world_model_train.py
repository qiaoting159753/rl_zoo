import torch
import logging
import numpy as np
from tqdm import trange
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm
import math

logging.basicConfig(level=logging.INFO)
from envs import DMCSEnvironment

# Agents
from agents.networks.mfrl.common import Actor
from agents.networks.mfrl.sac import SAC_Critic
from agents.mfrl import SAC
from utils import PrioritizedReplayBuffer

# World Models
from agents.networks.world_models.deterministic import (Probabilistic_Dynamics,
                                                        Gaussian_Process_World_Model,
                                                        Single_PNN,
                                                        NVP_World_Model,
                                                        Prior_World_Model,
                                                        Conditional_NVP_World_Model)

from agents.networks.world_models.ensembles import (Ensemble_Dyna_Ensemble_SAS_Reward,
                                                    Ensemble_NF_One_SAS_Reward,
                                                    Ensemble_Dyna_One_Reward)

from agents.networks.world_models.bayesian import (Bayesian_World_Model_BBB,
                                                   Bayesian_World_Model_Laplace_JA,
                                                   Bayesian_World_Model_Laplace_AX,
                                                   Bayesian_World_Model_SGLD_JA)


class World_Model_Trainer:
    """
    Training and evaluation loop for Model-Based agents that does not need to train the world model.
    """

    def __init__(self,
                 env: DMCSEnvironment,
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
        l2_multi_step_errors = []
        l1_one_step_errors = []
        l1_multi_step_errors = []

        l1_one_rwd_errors = []
        l1_multi_rwd_errors = []
        one_dyna_uncerts = []
        one_rwd_uncerts = []

        multi_dyna_uncerts = []
        multi_rwd_uncerts = []

        gt_s = self.env.reset()
        # multi_state = torch.FloatTensor(gt_s).to(self.device).unsqueeze(dim=0)
        episodic_pred_error = 0.0
        episodic_rwd_pred_error = 0.0

        for _ in range(self.episode_steps):
            if self.on_policy:
                action = self.agent.select_action_from_policy(gt_s)
            else:
                action = self.env.sample_action()
            gt_ns, gt_rwd, gt_done, _ = self.env.step(action)
            # Converting to tensor
            tensor_action = torch.FloatTensor(action).to(self.device).unsqueeze(dim=0)
            tensor_state = torch.FloatTensor(gt_s).to(self.device).unsqueeze(dim=0)
            # One step prediction
            pred_ns, _, _, _ = self.world_model.pred_next_states(observation=tensor_state,
                                                                 actions=tensor_action)
            pred_reward, _, _ = self.world_model.pred_rewards(observation=tensor_state,
                                                              action=tensor_action,
                                                              next_observation=pred_ns)
            # MSE. L1 of dynamics
            np_pred_ns = pred_ns.detach().squeeze().cpu().numpy()
            one_step_mse = (np.square(np_pred_ns - gt_ns)).mean()
            episodic_pred_error += one_step_mse
            one_step_l1 = (abs(np_pred_ns - gt_ns)).mean()
            l1_one_step_errors.append(one_step_l1)
            l2_one_step_errors.append(one_step_mse)

            pred_reward = pred_reward.detach().squeeze().cpu().numpy()
            l1_one_rwd_error = abs(pred_reward - gt_rwd)
            l1_one_rwd_errors.append(l1_one_rwd_error)

            one_dyna_uncert, one_rwd_uncert = self.world_model.estimate_uncertainty(observation=tensor_state,
                                                                                    actions=tensor_action)
            one_dyna_uncerts.append(one_dyna_uncert)
            one_rwd_uncerts.append(one_rwd_uncert)

            # Multi-step prediction with different actions.
            # if self.on_policy:
            #     np_multi_state = multi_state.detach().squeeze().cpu().numpy()
            #     multi_action = self.agent.select_action_from_policy(np_multi_state)
            # else:
            #     multi_action = action
            # multi_tensor_action = torch.FloatTensor(multi_action).to(self.device).unsqueeze(dim=0)
            # # Make accumulative multi-step predictions.
            # multi_state_pred, _, _, _ = self.world_model.pred_next_states(observation=multi_state,
            #                                                               actions=multi_tensor_action)
            # multi_pred_rewards, _, _ = self.world_model.pred_rewards(observation=multi_state,
            #                                                          action=multi_tensor_action,
            #                                                          next_observation=multi_state_pred)
            # np_multi_state = multi_state_pred.detach().squeeze().cpu().numpy()
            # multi_step_mse = (np.square(np_multi_state - gt_ns)).mean()
            # multi_step_l1 = (abs(np_multi_state - gt_ns)).mean()
            l1_multi_step_errors.append(0.0)
            l2_multi_step_errors.append(0.0)
            # L1 of Rewards
            # np_multi_pred_rewards = multi_pred_rewards.detach().squeeze().cpu().numpy()
            # episodic_rwd_pred_error += l1_one_rwd_error
            # l1_multi_rwd_error = abs(np_multi_pred_rewards - gt_rwd)
            # l1_multi_rwd_errors.append(l1_multi_rwd_error)
            l1_multi_rwd_errors.append(0.0)
            #################    Uncertainty Estimation and Quantification    ################
            # one_dyna_uncert, one_rwd_uncert = self.world_model.estimate_uncertainty(observation=multi_state,
            #                                                                         actions=multi_tensor_action)
            # multi_dyna_uncerts.append(multi_dyna_uncert)
            # multi_rwd_uncerts.append(multi_rwd_uncert)
            multi_dyna_uncerts.append(0.0)
            multi_rwd_uncerts.append(0.0)

            gt_s = gt_ns
            if gt_done:
                break

        l2_one_step_errors = np.array(l2_one_step_errors)
        l1_multi_step_errors = np.array(l1_multi_step_errors)
        l2_multi_step_errors = np.array(l2_multi_step_errors)
        l1_one_rwd_errors = np.array(l1_one_rwd_errors)
        l1_multi_rwd_errors = np.array(l1_multi_rwd_errors)
        one_dyna_uncerts = np.array(one_dyna_uncerts)
        one_rwd_uncerts = np.array(one_rwd_uncerts)
        multi_dyna_uncerts = np.array(multi_dyna_uncerts)
        multi_rwd_uncerts = np.array(multi_rwd_uncerts)

        c_1 = np.corrcoef(l2_one_step_errors, one_dyna_uncerts)
        c_2 = np.corrcoef(l1_one_step_errors, one_dyna_uncerts)
        # c_3 = np.corrcoef(l2_multi_step_errors, multi_dyna_uncerts)
        # c_4 = np.corrcoef(l1_multi_step_errors, multi_dyna_uncerts)
        c_5 = np.corrcoef(l1_one_rwd_errors, one_rwd_uncerts)
        # c_6 = np.corrcoef(l1_multi_rwd_errors, multi_rwd_uncerts)

        logging.info(f"Prediction Error: {episodic_pred_error}")
        all_data = np.zeros((9,))
        all_data[0] = self.counter
        all_data[1] = episodic_pred_error
        all_data[2] = episodic_rwd_pred_error
        if math.isnan(c_1[0, 1]):
            c_1[0, 1] = 0.0
        if math.isnan(c_2[0, 1]):
            c_2[0, 1] = 0.0
        all_data[3] = c_1[0, 1]
        all_data[4] = c_2[0, 1]
        # all_data[5] = c_3[0, 1]
        # all_data[6] = c_4[0, 1]
        all_data[7] = c_5[0, 1]
        # all_data[8] = c_6[0, 1]
        self.evaluation_array.append(all_data)

        if self.generate_results:
            # l2_one_step_errors = np.expand_dims(l2_one_step_errors, axis=0)
            # l1_one_step_errors = np.expand_dims(l1_one_step_errors, axis=0)
            # one_dyna_uncerts = np.expand_dims(one_dyna_uncerts, axis=0)
            # l2_multi_step_errors = np.expand_dims(l2_multi_step_errors, axis=0)
            # l1_multi_step_errors = np.expand_dims(l1_multi_step_errors, axis=0)
            # multi_dyna_uncerts = np.expand_dims(multi_dyna_uncerts, axis=0)
            # l1_one_rwd_errors = np.expand_dims(l1_one_rwd_errors, axis=0)
            # one_rwd_uncerts = np.expand_dims(one_rwd_uncerts, axis=0)
            # l1_multi_rwd_errors = np.expand_dims(l1_multi_rwd_errors, axis=0)
            # multi_rwd_uncerts = np.expand_dims(multi_rwd_uncerts, axis=0)
            # all_data = np.concatenate((l2_one_step_errors,
            #                            l1_one_step_errors,
            #                            one_dyna_uncerts,
            #                            l2_multi_step_errors,
            #                            l1_multi_step_errors,
            #                            multi_dyna_uncerts,
            #                            l1_one_rwd_errors,
            #                            one_rwd_uncerts,
            #                            l1_multi_rwd_errors,
            #                            multi_rwd_uncerts), axis=0)
            # Save the metrics
            file_name = self.directory + str(self.seed) + "_" + self.date_and_time + ".csv"
            np.savetxt(file_name + ".csv", np.array(self.evaluation_array), delimiter=",")
            # with open(file_name, 'ab') as fff:
            #     np.savetxt(fff, all_data, delimiter=",")
            # fff.close()

    def train(self, flush=False):
        """
        Train the MFRL Agent.
        """
        with logging_redirect_tqdm():
            need_evaluate = False
            need_reset = True
            if flush:
                store_states = []
                store_actions = []
                store_next_states = []

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
                if flush:
                    store_states.append(state)
                    store_actions.append(action)
                    store_next_states.append(next_state)

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
                            if flush:
                                tensor_store_states = torch.FloatTensor(np.array(store_states)).to(self.device)
                                tensor_store_actions = torch.FloatTensor(np.array(store_actions)).to(self.device)
                                tensor_store_next_states = torch.FloatTensor(np.array(store_next_states)).to(
                                    self.device)
                                self.world_model.train_world_all(tensor_store_states, tensor_store_actions,
                                                                 tensor_store_next_states)
                            else:
                                self.agent.train_world_model(memory=self.memory,
                                                             batch_size=self.batch_size,
                                                             world_model=self.world_model)
                    else:
                        # For every a few steps
                        if self.counter % (int(1.0 / self.model_G)) == 0:
                            if flush:
                                tensor_store_states = torch.FloatTensor(np.array(store_states)).to(self.device)
                                tensor_store_actions = torch.FloatTensor(np.array(store_actions)).to(self.device)
                                tensor_store_next_states = torch.FloatTensor(np.array(store_next_states)).to(
                                    self.device)
                                self.world_model.train_world_all(tensor_store_states, tensor_store_actions,
                                                                 tensor_store_next_states)
                            else:
                                self.agent.train_world_model(memory=self.memory,
                                                             batch_size=self.batch_size,
                                                             world_model=self.world_model)
                # Evaluating
                if self.counter % self.evaluate_interval == 0:
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
        self.agent = SAC(actor_network=actor,
                         critic_network=critic,
                         action_num=self.action_dim,
                         alpha_lr=3e-4,
                         gamma=0.99,
                         tau=0.005,
                         actor_lr=3e-4,
                         critic_lr=3e-4,
                         device=torch.device(self.device),
                         reward_scale=1.0)

        if self.world_model_name == "Prior_World_Model":
            self.world_model = Prior_World_Model(observation_size=self.state_dim,
                                                 num_actions=self.action_dim,
                                                 l_r=0.0001,
                                                 device=self.device, hidden_size=256)

        if self.world_model_name == "Single_PNN":
            self.world_model = Single_PNN(observation_size=self.state_dim, num_actions=self.action_dim, l_r=0.001,
                                          device=self.device)

        if self.world_model_name == "Ensemble_Dyna_One_Reward":
            self.world_model = Ensemble_Dyna_One_Reward(observation_size=self.state_dim,
                                                        num_actions=self.action_dim,
                                                        num_models=5,
                                                        l_r=0.001,
                                                        device=self.device,
                                                        boost_inter=int(self.parameter_a),
                                                        sas=self.sas,
                                                        prob_rwd=self.prob_rwd)
        if self.world_model_name == "Bayesian_VI":
            self.world_model = Bayesian_World_Model_BBB(observation_size=self.state_dim, num_actions=self.action_dim,
                                                        l_r=0.001, device=self.device, ratio=self.parameter_a,
                                                        sigma=self.parameter_b)

        if self.world_model_name == "Bayesian_Laplace_AX":
            self.world_model = Bayesian_World_Model_Laplace_AX(observation_size=self.state_dim,
                                                               num_actions=self.action_dim,
                                                               l_r=0.001,
                                                               hidden_size=256,
                                                               device=self.device,
                                                               sigma=self.parameter_a,
                                                               temperature=self.parameter_b,
                                                               prior_precision=self.parameter_c)

        if self.world_model_name == "Bayesian_World_Model_SGLD_JA":
            self.world_model = Bayesian_World_Model_SGLD_JA(observation_size=self.state_dim,
                                                            num_actions=self.action_dim,
                                                            l_r=0.001,
                                                            hidden_size=256,
                                                            device=self.device)

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

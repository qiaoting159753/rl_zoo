import pandas as pd
import torch
import logging
import numpy as np
from tqdm import trange
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm
from rl_zoo.utils import normalize
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# Agents
from rl_zoo.networks.mfrl.common import Actor
from rl_zoo.networks.mfrl.sac import SAC_Critic
from rl_zoo.agents.mbrl import Dyna_SAC_NS, Immerseive_Weighting_Dyna_SAC_NS
from rl_zoo.utils import PrioritizedReplayBuffer
from rl_zoo.networks.world_models.ensembles import Ensemble_Dyna_One_Reward, Ensemble_Dyna_Big


class MBRL_Trainer:
    """
    Training and evaluation loop for Model-Based agents that does not need to train the world model.
    """

    def __init__(self,
                 env,
                 mbrl_agent_name: str,
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
                 gripper: bool,
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
        self.gripper = gripper
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
        self.training_rewards = 0.0

        # Agent and world model.
        self.mbrl_agent_name = mbrl_agent_name
        self.agent = None
        self.world_model = None
        self.on_policy = on_policy
        self.memory = PrioritizedReplayBuffer()
        self.device = device
        self.agent_selection()

        logging.info(f"Name: {self.mbrl_agent_name}, On Policy: {on_policy}, Random Goal:{random_goal}")

    def evaluate(self):
        """
        Evaluation
        """
        # Evaluate 10 times
        l1_one_rwd_errors = []
        gt_s = self.env.reset()
        episodic_pred_error = 0.0
        episodic_rwd_pred_error = 0.0
        epi_total_rwd = 0.0
        for _ in range(self.episode_steps):
            if self.on_policy:
                action = self.agent.select_action_from_policy(gt_s)
            else:
                action = self.env.sample_action()
            action = normalize(action, self.env.max_action_value, self.env.min_action_value)
            gt_ns, gt_rwd, gt_done, _ = self.env.step(action)
            epi_total_rwd += gt_rwd
            # Converting to tensor
            tensor_action = torch.FloatTensor(action).to(self.device).unsqueeze(dim=0)
            tensor_state = torch.FloatTensor(gt_s).to(self.device).unsqueeze(dim=0)
            # One step prediction
            pred_ns, _, _, _ = self.world_model.pred_next_states(observation=tensor_state,
                                                                 actions=tensor_action)
            # MSE. L1 of dynamics
            np_pred_ns = pred_ns.detach().squeeze().cpu().numpy()
            one_step_mse = (np.square(np_pred_ns - gt_ns)).mean()
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
            #################    Uncertainty Estimation and Quantification    ################
            gt_s = gt_ns
            if gt_done:
                break
        logging.info(
            f"Dyna Err: {episodic_pred_error}, Rwd Err:{episodic_rwd_pred_error}, Train rwd:{self.training_rewards}, Eval rwd:{epi_total_rwd}")

        # Counter, Dynamics Error, Reward Error. Eps rewards.
        all_data = np.zeros((5,))
        all_data[0] = self.counter
        all_data[1] = episodic_pred_error
        all_data[2] = episodic_rwd_pred_error
        all_data[3] = self.training_rewards
        all_data[4] = epi_total_rwd
        self.evaluation_array.append(all_data)

        if self.generate_results:
            # Save the metrics
            data_result = np.array(self.evaluation_array)
            file_name = self.directory + str(self.seed) + ".csv"
            np.savetxt(file_name + ".csv", np.array(self.evaluation_array), delimiter=",")

            pnn_mean = pd.DataFrame(data={'steps': data_result[:, 0].astype(int), 'Data': data_result[:, 4]})
            sns.lineplot(
                data=pnn_mean,
                x=pnn_mean["steps"],
                y="Data",
                label="pnn",
                errorbar="sd",
            )
            plt.ylim(0, 1000)
            plt.style.use("seaborn-v0_8")
            label_fontsize = 15
            ticks_fontsize = 10
            plt.grid()
            plt.xticks(fontsize=ticks_fontsize)
            plt.yticks(fontsize=ticks_fontsize)
            plt.xlabel("Steps", fontsize=label_fontsize)
            plt.ylabel("Rewards", fontsize=label_fontsize)
            plt.legend(loc="best").set_draggable(True)
            plt.tight_layout(pad=0.5)
            plt.savefig(self.directory + self.mbrl_agent_name + "_eval_rwds" + ".png")
            plt.close()

    def train(self, flush=False):
        """
        Train the MFRL Agent.
        """
        with logging_redirect_tqdm():
            need_evaluate = False
            need_reset = True
            for _ in trange(self.maximum_steps):
                if need_reset:
                    self.training_rewards = 0.0
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
                self.training_rewards += reward
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
        self.world_model = Ensemble_Dyna_Big(observation_size=self.state_dim,
                                             num_actions=self.action_dim,
                                             device=self.device,
                                             boost_inter=int(self.parameter_a),
                                             num_models=int(self.parameter_b),
                                             sas=self.sas,
                                             prob_rwd=self.prob_rwd)

        actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
        critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)

        if self.mbrl_agent_name == "Dyna_SAC_NS":
            self.agent = Dyna_SAC_NS(actor_network=actor,
                                     critic_network=critic,
                                     world_network=self.world_model,
                                     action_num=self.action_dim,
                                     alpha_lr=3e-4,
                                     horizon=1,
                                     num_samples=10,
                                     gamma=0.99,
                                     tau=0.005,
                                     actor_lr=3e-4,
                                     critic_lr=3e-4,
                                     train_reward=self.train_reward,
                                     train_both=self.train_both,
                                     device=torch.device(self.device),
                                     gripper=self.gripper)

        if self.mbrl_agent_name == "Dyna_SAC_Immersive_Weighting":
            self.agent = Immerseive_Weighting_Dyna_SAC_NS(actor_network=actor,
                                                          critic_network=critic,
                                                          world_network=self.world_model,
                                                          action_num=self.action_dim,
                                                          alpha_lr=3e-4,
                                                          gamma=0.99,
                                                          tau=0.005,
                                                          actor_lr=3e-4,
                                                          critic_lr=3e-4,
                                                          horizon=1,
                                                          num_samples=10,
                                                          threshold=self.parameter_c,
                                                          device=torch.device(self.device),
                                                          train_reward=self.train_reward,
                                                          train_both=self.train_both,
                                                          gripper=self.gripper)

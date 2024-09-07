import os
import torch
import logging
import numpy as np
from tqdm import trange
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm
logging.basicConfig(level=logging.INFO)

# Agents
from agents.networks.mfrl.common import Actor
from agents.networks.mfrl.sac import SAC_Critic
from agents.mfrl import SAC
from utils import PrioritizedReplayBuffer

from envs import DMCSEnvironment

# World Models
from agents.networks.world_models.ensembles import Ensemble_Dyna_One_SAS_Reward
from agents.networks.world_models.deterministic import Probabilistic_Dynamics
from agents.networks.world_models.ensembles import Ensemble_Dyna_One_NS_Reward
from agents.networks.world_models.ensembles import Ensemble_Dyna_Ensemble_SAS_Reward
from agents.networks.world_models.ensembles import En


class MBRL_Trainer:
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
                 seed: int):
        self.world_model_name = world_model_name
        self.on_policy = on_policy
        self.seed = seed
        self.generate_results = generate_results
        self.maximum_steps = maximum_steps
        self.episode_steps = episode_steps
        self.env = env
        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')

        self.evaluate_interval = evaluate_interval
        self.evaluation_array = []

        self.counter = 0
        self.device = device
        self.random_goal = random_goal
        self.G = G
        self.model_G = model_G
        self.batch_size = batch_size

        self.state_dim = env.observation_space
        self.action_dim = env.action_num

        self.agent = None
        self.memory = PrioritizedReplayBuffer()

        self.directory = "/root/rl_zoo_data/"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def evaluate(self):
        """
        Evaluate the agents
        """
        record_rewards = np.zeros((11,))
        record_rewards[10] = self.counter  # Index
        for j in range(10):
            s = self.env.reset()
            total_rewards = 0.0
            for _ in range(self.episode_steps):
                a = self.agent.select_action_from_policy(s, evaluation=True)
                ns, rwd, done, _ = self.env.step(a)
                total_rewards += rwd
                s = ns
                if done:
                    break
            record_rewards[j] = total_rewards
        self.evaluation_array.append(record_rewards)
        logging.info(f"--Evaluation ({self.counter}/{self.maximum_steps}): " + str(np.mean(record_rewards[:10])) + "--")
        if self.generate_results:
            eval_array = np.array(self.evaluation_array)
            data_folder = self.directory
            # Save the metrics
            file_name = data_folder + str(self.seed) + "_" + self.env.domain + "_" + \
                        self.env.task + "_" + self.agent_name + "_" + self.date_and_time
            np.savetxt(file_name + ".csv", eval_array, delimiter=",")

    def train(self):
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
                state = next_state
                step_counter += 1
                epi_reward += reward
                if len(self.memory) > self.batch_size:
                    for _ in range(self.G):
                        self.agent.train_policy(self.memory, batch_size=self.batch_size)
                    self.counter += 1

                    if self.model_G > 1.0:
                        for _ in range(int(self.model_G)):
                            # self.agent.train_world_model()
                            print("Train world model")
                    else:
                        # For every a few steps
                        if self.counter % (int(1.0/self.model_G)) == 0:
                            # self.agent.train_world_model()
                            print("Train world model")

                if self.counter % self.evaluate_interval == 0:
                    need_evaluate = True
                if done or ((step_counter % self.episode_steps) == 0):
                    logging.info("Training: " + str(epi_reward))
                    need_reset = True
                    if need_evaluate:
                        self.evaluate()
                        need_evaluate = False

    def agent_selection(self, agent_name):
        """
        Create an agent

        :param agent_name:
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

        if self.world_model_name == "Ensemble_Dyna_One_SAS_Reward":
            self.world_model = Ensemble_Dyna_One_SAS_Reward(observation_size=self.state_dim,
                                                           num_actions=self.action_dim,
                                                           num_models=5,
                                                           lr=0.001,
                                                           device=self.device)

        if self.world_model_name == "Ensemble_Dyna_One_SAS_Reward":
            self.world_model = Ensemble_Dyna_One_SAS_Reward(observation_size=self.state_dim,
                                                           num_actions=self.action_dim,
                                                           num_models=5,
                                                           lr=0.001,
                                                           device=self.device)
        if self.world_model_name == "Ensemble_Dyna_One_SAS_Reward":
            self.world_model = Ensemble_Dyna_One_SAS_Reward(observation_size=self.state_dim,
                                                           num_actions=self.action_dim,
                                                           num_models=5,
                                                           lr=0.001,
                                                           device=self.device)

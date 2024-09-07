import os
import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.INFO)

# Agents
from agents.mbrl import Dyna_SAC_NS
from agents.mbrl import Dyna_SAC_SAS
from agents.mbrl import Immersive_Reweight_Dyna_SAC
from agents.mbrl import STEVE_SAC_Critic_mean

# World Models
from agents.networks.world_models.ensembles import Ensemble_Dyna_One_SAS_Reward
from agents.networks.world_models.ensembles import Ensemble_Dyna_One_NS_Reward

# Actors and Critics
from agents.networks.mfrl.common import Actor
from agents.networks.mfrl.sac import SAC_Critic
from agents.networks.mfrl.tqc import TQC_Critic

from utils import PrioritizedReplayBuffer
from datetime import datetime
import numpy as np

from envs import DMCSEnvironment


class MBRL_Trainer:
    """
    Training and evaluation loop for Model-Based agents that does not need to train the world model.
    """

    def __init__(self,
                 env: DMCSEnvironment,
                 agent_name: str,
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
        self.agent_name = agent_name
        self.agent_selection(agent_name)
        self.memory = PrioritizedReplayBuffer()

        # self.directory = "/root/rl_zoo_data/"
        self.directory = "statistic/"
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
                action = self.agent.select_action_from_policy(state)
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
        sas_world_model = Ensemble_Dyna_One_SAS_Reward(observation_size=self.state_dim,
                                                       num_actions=self.action_dim,
                                                       num_models=5,
                                                       lr=0.001,
                                                       device=self.device)

        if agent_name == "Dyna_SAC_NS":
            sac_critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
            ns_world_model = Ensemble_Dyna_One_NS_Reward(observation_size=self.state_dim,
                                                         num_actions=self.action_dim,
                                                         num_models=5,
                                                         lr=0.001,
                                                         device=self.device)

            self.agent = Dyna_SAC_NS(
                actor_network=actor,
                critic_network=sac_critic,
                world_network=ns_world_model,
                gamma=0.99,
                tau=0.005,
                action_num=self.action_dim,
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                num_samples=self.brach_factor,
                horizon=self.horizon
            )

        if agent_name == "Dyna_SAC_SAS":
            sac_critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
            self.agent = Dyna_SAC_SAS(
                actor_network=actor,
                critic_network=sac_critic,
                world_network=sas_world_model,
                gamma=0.99,
                tau=0.005,
                action_num=self.action_dim,
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                num_samples=self.brach_factor,
                horizon=self.horizon
            )

        if agent_name == "Immersive_Reweight_Dyna_SAC":
            sac_critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
            self.agent = Immersive_Reweight_Dyna_SAC(
                actor_network=actor,
                critic_network=sac_critic,
                world_network=sas_world_model,
                gamma=0.99,
                tau=0.005,
                action_num=self.action_dim,
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                num_samples=self.brach_factor,
                horizon=self.horizon
            )

        if agent_name == "STEVE_SAC_Critic_mean":
            sac_critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
            self.agent = STEVE_SAC_Critic_mean(
                actor_network=actor,
                critic_network=sac_critic,
                world_network=sas_world_model,
                gamma=0.99,
                tau=0.005,
                action_num=self.action_dim,
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                horizon=self.horizon
            )

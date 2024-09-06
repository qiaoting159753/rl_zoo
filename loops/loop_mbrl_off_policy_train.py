import torch

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
                 model_G: int,
                 batch_size: int,
                 episode_steps: int,
                 maximum_steps: int,
                 horizon: int,
                 branch_factor: int,
                 ):

        self.horizon = horizon
        self.brach_factor = branch_factor

        self.maximum_steps = maximum_steps
        self.episode_steps = episode_steps
        self.model_G = model_G

        self.agent = None
        self.agent_name = agent_name

        self.env = env
        self.agent_selection(agent_name)

        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        self.evaluation_array = [[], [], [], [], []]
        self.generate_results = True
        self.counter = 0

        self.device = device
        self.random_goal = random_goal
        self.G = G
        self.batch_size = batch_size

        self.state_dim = self.env.observation_space
        self.action_dim = self.env.action_num

        self.memory = PrioritizedReplayBuffer()

    def evaluate(self):
        """
        Evaluate the MBRL agent

        """
        total_rewards = 0.0
        dones = 0
        total_dist = 0.0
        total_steps = 0
        total_qs = 0.0

        for _ in range(10):
            state = self.env.reset()
            for _ in range(self.episode_steps):
                action = self.agent.select_action_from_policy(state, evaluation=True)
                next_state, reward, done, dist, _ = self.env.step(action)
                total_dist += dist
                total_steps += 1
                total_rewards += reward
                state = next_state
                if done:
                    dones += 1
                    break
        avg_reward = total_rewards / total_steps
        print("------ Evaluation: " + str(total_rewards / 10) + " ------")
        self.evaluation_array[0].append(self.counter)
        self.evaluation_array[1].append(avg_reward)
        self.evaluation_array[2].append(dones)
        self.evaluation_array[3].append(total_dist)
        self.evaluation_array[4].append(total_qs / total_steps)

        if self.generate_results:
            eval_array = np.array(self.evaluation_array)
            # Save the metrics
            file_name = self.env.domain + "_" + self.env.task + "_" + self.agent_name + "_" + self.date_and_time
            np.savetxt(file_name + ".csv", eval_array, delimiter=",")
            self.agent.save_models(file_name)

    def train(self):
        """
        Train the MBRL Agent
        """
        for i in range(self.maximum_steps):
            state = self.env.reset()
            epi_reward = 0.0
            for _ in range(self.episode_steps):
                action = self.agent.select_action_from_policy(state)
                # Do action is for the environment.
                next_state, reward, done, _, _ = self.env.step(action)
                # Small action is for training.
                self.memory.add(state, action, reward, next_state, done)
                epi_reward += reward
                if len(self.memory) > self.batch_size:
                    for _ in range(self.G):
                        self.agent.train_policy(self.memory, batch_size=self.batch_size)
                        self.counter += 1
                state = next_state
                if done:
                    break
            # Print Training Rewards.
            print("------ Training: " + str(epi_reward) + " ------")
            if i % 10000 == 0:
                self.evaluate()

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
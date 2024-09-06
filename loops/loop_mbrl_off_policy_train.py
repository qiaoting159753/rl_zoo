import torch

# Agents
from agents.mbrl import Dyna_SAC_NS
from agents.mbrl import Dyna_SAC_SAS
from agents.mbrl import Dyna_TQC
from agents.mbrl import Immersive_Reweight_Dyna_SAC
from agents.mbrl import Immersive_Reweight_Dyna_TQC

from agents.mbrl import STEVE_SAC_actor
from agents.mbrl import STEVE_SAC_Critic_all
from agents.mbrl import STEVE_SAC_Critic_mean
from agents.mbrl import STEVE_TQC_Actor
from agents.mbrl import STEVE_TQC_Critic


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
                 branch_factor:int,
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
        sac_critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
        tqc_critic = TQC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)

        action_dim = self.action_dim,
        state_dim = self.state_dim,
        actor_lr = 3e-4,
        critic_lr = 3e-4,

        if agent_name == "Dyna_SAC_NS":
            Dyna_SAC_NS()
            self.agent = Dyna_SAC_NS(actor_network = actor, critic_network = sac_critic,
                world_network: EnsembleWorldAndOneNSReward,
                gamma = 0.99,
                tau: 0.005,
                action_num: int,
                actor_lr: float,
                critic_lr: float,
                alpha_lr = 3e-4,
                num_samples = self.brach_factor,
                horizon = self.horizon,
                device = self.device,
            )
            def __init__(
                    self,
                    actor_network: torch.nn.Module,
                    critic_network: torch.nn.Module,
                    world_network: EnsembleWorldAndOneNSReward,
                    gamma: float,
                    tau: float,
                    action_num: int,
                    actor_lr: float,
                    critic_lr: float,
                    alpha_lr: float,
                    num_samples: int,
                    horizon: int,
                    device: torch.device,
            ):

        if agent_name == "Dyna_SAC_SAS":
            print("Dyna")

        if agent_name == "Dyna_TQC":
            print("Dyna")

        if agent_name == "Immersive_Reweight_Dyna_SAC":
            print("Dyna")

        if agent_name == "Immersive_Reweight_Dyna_TQC":
            print("Dyna")

        if agent_name == "STEVE_SAC_actor":
            print("Dyna")

        if agent_name == "STEVE_SAC_critic_all":
            print("Dyna")

        if agent_name == "STEVE_SAC_critic_mean":
            print("Dyna")

        if agent_name == "STEVE_TQC_actor":
            print("Dyna")

        if agent_name == "STEVE_TQC_critic":
            print("Dyna")




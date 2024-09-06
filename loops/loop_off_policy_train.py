# Agents
import torch
import logging

from agents.mfrl import SAC
from agents.mfrl import TQC
from agents.mfrl import Fully_Expand

# Actors
from agents.networks.mfrl.common import Actor
from agents.networks.mfrl.common import HyperActor

# Critic
from agents.networks.mfrl.sac import SAC_Critic
from agents.networks.mfrl.sac import Hyper_Double_SAC_Critic
from agents.networks.mfrl.tqc import TQC_Critic
from agents.networks.mfrl.tqc import Hyper_TQC_Critic

from utils import PrioritizedReplayBuffer
from datetime import datetime
import numpy as np

from envs import DMCSEnvironment


class MFRL_Trainer:
    """
    Training and evaluation loop for Model-Free agents that does not need to train the world model.
    """

    def __init__(self,
                 env: DMCSEnvironment,
                 agent_name: str,
                 random_goal: bool,
                 device: str,
                 G: int,
                 batch_size: int,
                 episode_steps: int,
                 maximum_steps: int,
                 evaluate_interval: int,
                 generate_results: bool):

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
        self.batch_size = batch_size

        self.state_dim = env.observation_space
        self.action_dim = env.action_num

        self.agent = None
        self.agent_name = agent_name
        self.agent_selection(agent_name)
        self.memory = PrioritizedReplayBuffer()

    def evaluate(self):
        """
        Evaluate the agents
        """
        total_steps = 0
        record_rewards = np.zeros((11,))
        for j in range(10):
            state = self.env.reset()
            total_rewards = 0.0
            for _ in range(self.episode_steps):
                action = self.agent.select_action_from_policy(state, evaluation=True)
                next_state, reward, done, _ = self.env.step(action)
                total_steps += 1
                total_rewards += reward
                state = next_state
                if done:
                    break
            record_rewards[j] = total_rewards
        record_rewards[10] = self.counter  # Index
        self.evaluation_array.append(record_rewards)
        logging.info("------------------ Evaluation: " + str(np.mean(record_rewards[:10])) + " ------------------")
        if self.generate_results:
            eval_array = np.array(self.evaluation_array)
            logging.info(eval_array.shape)
            data_folder = "statistics/"
            # Save the metrics
            file_name = data_folder + self.env.domain + "_" + \
                        self.env.task + "_" + self.agent_name + "_" + self.date_and_time
            np.savetxt(file_name + ".csv", eval_array, delimiter=",")

    def train(self):
        """
        Train the MFRL Agent.
        """
        need_evaluate = False
        for _ in range(self.maximum_steps):
            state = self.env.reset()
            epi_reward = 0.0
            for _ in range(self.episode_steps):
                action = self.agent.select_action_from_policy(state)
                # Do action is for the environment.
                next_state, reward, done, _ = self.env.step(action)
                # Small action is for training.
                self.memory.add(state, action, reward, next_state, done)
                epi_reward += reward
                if len(self.memory) > self.batch_size:
                    for _ in range(self.G):
                        self.agent.train_policy(self.memory, batch_size=self.batch_size)
                self.counter += 1
                if self.counter % self.evaluate_interval == 0:
                    need_evaluate = True
                state = next_state
                if done:
                    break
            print("---- Training: " + str(epi_reward) + " ----")
            if need_evaluate:
                self.evaluate()
                need_evaluate = False

    def agent_selection(self, agent_name):
        """
        Create an agent
        :param agent_name:
        """
        if agent_name == "SAC":
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

        if agent_name == "TQC":
            actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
            critic = TQC_Critic(observation_size=self.state_dim,
                                num_actions=self.action_dim,
                                num_quantiles=25,
                                num_critics=5)
            self.agent = TQC(actor_network=actor,
                             critic_network=critic,
                             actor_lr=3e-4,
                             critic_lr=3e-4,
                             alpha_lr=3e-4,
                             gamma=0.99,
                             tau=0.005,
                             top_quantiles_to_drop=2,
                             action_num=self.action_dim,
                             device=self.device
                             )

        if agent_name == "Fully_Expand":
            actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
            self.agent = Fully_Expand(self,
                                      actor_network=actor,
                                      gamma=0.99,
                                      tau=0.005,
                                      action_num=6,
                                      actor_lr=0.0003,
                                      alpha_lr=0.0003,
                                      horizon=10,
                                      device='cpu')

        if agent_name == "Hyper_SAC_Critic":
            actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
            critic = Hyper_Double_SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
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

        if agent_name == "Hyper_TQC_Critic":
            actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
            critic = Hyper_TQC_Critic(observation_size=self.state_dim,
                                      num_actions=self.action_dim,
                                      num_quantiles=25,
                                      num_critics=5)
            self.agent = TQC(actor_network=actor,
                             critic_network=critic,
                             actor_lr=3e-4,
                             critic_lr=3e-4,
                             alpha_lr=3e-4,
                             gamma=0.99,
                             tau=0.005,
                             top_quantiles_to_drop=2,
                             action_num=self.action_dim,
                             device=self.device)

        if agent_name == "Hyper_SAC_all":
            actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
            critic = Hyper_Double_SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
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

        if agent_name == "Hyper_TQC_all":
            actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
            critic = Hyper_TQC_Critic(observation_size=self.state_dim,
                                      num_actions=self.action_dim,
                                      num_quantiles=25,
                                      num_critics=5)
            self.agent = TQC(actor_network=actor,
                             critic_network=critic,
                             actor_lr=3e-4,
                             critic_lr=3e-4,
                             alpha_lr=3e-4,
                             gamma=0.99,
                             tau=0.005,
                             top_quantiles_to_drop=2,
                             action_num=self.action_dim,
                             device=self.device)

        if agent_name == "Hyper_SAC_actor":
            actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
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

        if agent_name == "Hyper_TQC_actor":
            actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
            critic = TQC_Critic(observation_size=self.state_dim,
                                num_actions=self.action_dim,
                                num_quantiles=25,
                                num_critics=5)
            self.agent = TQC(actor_network=actor,
                             critic_network=critic,
                             actor_lr=3e-4,
                             critic_lr=3e-4,
                             alpha_lr=3e-4,
                             gamma=0.99,
                             tau=0.005,
                             top_quantiles_to_drop=2,
                             action_num=self.action_dim,
                             device=self.device)

import torch
# Agents
from agents.mbrl import Dyna_SAC
from agents.mbrl import Immersive_Reweight_Dyna_SAC

# Actors and Critics
from agents.networks.mfrl.common import Actor
from agents.networks.mfrl.sac import SAC_Critic
from agents.networks.mfrl.tqc import TQC_Critic

from utils import MemoryBuffer
from datetime import datetime
import numpy as np

from envs import DMCSEnvironment

# Dyna_SAC
# Dyna_TQC
# Dyna_SAC_Immersive_Reweight
# Dyna_TQC_Immersive_Reweight
# MVE_SAC_mean
# MVE_SAC_all
# MVE_TQC_mean
# MVE_TQC_all



class MBRL_Trainer:
    """
    Training and evaluation loop for Model-Based agents that does not need to train the world model.
    """

    def __init__(self, env: DMCSEnvironment, agent_name, random_goal, device, G, batch_size):
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

        self.memory = MemoryBuffer()

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
            for _ in range(10):
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
        print("-----------Evaluation: " + str(avg_reward))
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
        for i in range(1000000):
            state = self.env.reset()
            epi_reward = 0.0
            for _ in range(10):
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
            if i % 100 == 0:
                self.evaluate()

    def agent_selection(self, agent_name):
        """
        Create an agent

        :param agent_name:
        """
        actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
        sac_critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
        tqc_critic = TQC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)


        if agent_name == "Dyna_SAC":
            print("Dyna")

        if agent_name == "Dyna_TQC":
            print("Dyna")

        if agent_name == "Dyna_SAC_Immersive_Reweight":
            print("Dyna")

        if agent_name == "Dyna_SAC_Immersive_Reweight":
            print("Dyna")

        if agent_name == "Dyna_SAC_Immersive_Reweight":
            print("Dyna")






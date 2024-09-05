import torch
# Agents
from agents.tqc.agent_tqc import TQC
from agents.tqc import ME_TQC
from agents.sac import SAC
from agents.sac import ME_SAC
from agents.fully_expand import Fully_Expand

# Actors
from agents.common import Actor
from agents.common import HyperActor
# Critic
from agents.tqc import TQC_Critic
from agents.tqc import Hyper_TQC_Critic

from agents.memory_buffer import MemoryBuffer
from datetime import datetime
import numpy as np


class Trainer:
    def __init__(self, env, action_dim):
        self.agent = None
        self.agent_selection()

        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        self.evaluation_array = [[], [], [], [], []]
        self.generate_results = True
        self.random_goal = False

        self.G = 3
        self.batch_size = 256

        self.counter = 0
        self.env = env
        self.device = "cpu"

        self.state_dim = 20
        self.action_dim = action_dim

        self.memory = MemoryBuffer()

    def evaluate(self):
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
            file_name = "data/Kinematic" + "_" + self.date_and_time
            np.savetxt(file_name + "_eval_rewards.csv", eval_array, delimiter=",")
            self.agent.save_models("low_cost")

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

    def agent_selection(self):
        # state_dim: 6, action_dim: 3
        # actor = HyperActor(observation_size=self.state_dim, num_actions=self.do_action_dim)
        actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
        self.agent = Fully_Expand(self,
                                  env=self.env,
                                  actor_network=actor,
                                  gamma=0.99,
                                  tau=0.005,
                                  action_num=6,
                                  actor_lr=0.0003,
                                  alpha_lr=0.0003,
                                  horizon=10,
                                  device='cpu')

        # self.agent = TQC(actor_network=actor,
        #                  critic_network=critic,
        #                  actor_lr=3e-4,
        #                  critic_lr=3e-4,
        #                  alpha_lr=3e-4,
        #                  gamma=0.99,
        #                  tau=0.005,
        #                  top_quantiles_to_drop=2,
        #                  action_num=self.action_dim,
        #                  device=self.device,)

        # self.agent = METQC(env=self.env,
        #                    actor_network=actor,
        #                    critic_network=critic,
        #                    actor_lr=3e-4,
        #                    critic_lr=3e-4,
        #                    alpha_lr=3e-4,
        #                    gamma=0.99,
        #                    tau=0.005,
        #                    top_quantiles_to_drop=2,
        #                    action_num=self.action_dim,
        #                    device=self.device, )

        # self.agent = SAC(actor_network=actor,
        #                  critic_network=critic,
        #                  alpha_lr=3e-4,
        #                  gamma=0.99,
        #                  tau=0.005,
        #                  reward_scale=1.0,
        #                  action_num=self.do_action_dim,
        #                  actor_lr=3e-4,
        #                  critic_lr=3e-4,
        #                  device=self.device)

        # self.agent = ME_SAC(env=env,
        #                     actor_network=actor,
        #                     critic_network=critic,
        #                     alpha_lr=3e-4,
        #                     gamma=0.99,
        #                     tau=0.005,
        #                     reward_scale=1.0,
        #                     action_num=self.do_action_dim,
        #                     actor_lr=3e-4,
        #                     critic_lr=3e-4,
        #                     device=self.device)

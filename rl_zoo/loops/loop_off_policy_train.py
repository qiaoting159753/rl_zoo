# # Agents
# import torch
# import logging
# from tqdm import trange
# from tqdm.contrib.logging import logging_redirect_tqdm
# logging.basicConfig(level=logging.INFO)

# from rl_zoo.agents.mfrl import SAC
# from rl_zoo.agents.mfrl import TQC
# from rl_zoo.agents.mfrl import Fully_Expand

# # Actors
# from rl_zoo.networks.mfrl.common import Actor
# from rl_zoo.networks.mfrl.common import HyperActor

# # Critic
# from rl_zoo.networks.mfrl.sac import SAC_Critic
# from rl_zoo.networks.mfrl.sac import Hyper_Double_SAC_Critic
# from rl_zoo.networks.mfrl.tqc import TQC_Critic
# from rl_zoo.networks.mfrl.tqc import Hyper_TQC_Critic

# from rl_zoo.utils import PrioritizedReplayBuffer
# from datetime import datetime
# import numpy as np

# from rl_zoo.envs import DMCSEnvironment


# class MFRL_Trainer:
#     """
#     Training and evaluation loop for Model-Free agents that does not need to train the world model.
#     """

#     def __init__(self,
#                  env: DMCSEnvironment,
#                  agent_name: str,
#                  random_goal: bool,
#                  device: str,
#                  G: int,
#                  batch_size: int,
#                  episode_steps: int,
#                  maximum_steps: int,
#                  evaluate_interval: int,
#                  generate_results: bool,
#                  seed: int,
#                  directory):
#         # Environment
#         self.env = env
#         self.random_goal = random_goal
#         self.state_dim = env.observation_space
#         self.action_dim = env.action_num

#         # Agents
#         self.device = device
#         self.agent = None
#         self.agent_name = agent_name
#         self.agent_selection(agent_name)
#         self.memory = PrioritizedReplayBuffer()

#         # Training
#         self.seed = seed
#         self.maximum_steps = maximum_steps
#         self.episode_steps = episode_steps
#         self.G = G
#         self.batch_size = batch_size
#         self.counter = 0
#         self.evaluate_interval = evaluate_interval

#         # Save Data
#         self.evaluation_array = []
#         self.directory = directory
#         self.generate_results = generate_results
#         self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')

#     def evaluate(self):
#         """
#         Evaluate the agents
#         """
#         record_rewards = np.zeros((11,))
#         record_rewards[10] = self.counter  # Index
#         for j in range(10):
#             s = self.env.reset()
#             total_rewards = 0.0
#             for _ in range(self.episode_steps):
#                 a = self.agent.select_action_from_policy(s, evaluation=True)
#                 ns, rwd, done, _ = self.env.step(a)
#                 total_rewards += rwd
#                 s = ns
#                 if done:
#                     break
#             record_rewards[j] = total_rewards
#         self.evaluation_array.append(record_rewards)
#         logging.info(f"--Evaluation ({self.counter}/{self.maximum_steps}): " + str(np.mean(record_rewards[:10])) + "--")
#         if self.generate_results:
#             eval_array = np.array(self.evaluation_array)
#             # Save the metrics
#             file_name = self.directory + str(self.seed) + "_" + self.env.domain + "_" + \
#                         self.env.task + "_" + self.agent_name + "_" + self.date_and_time
#             np.savetxt(file_name + ".csv", eval_array, delimiter=",")

#     def train(self):
#         """
#         Train the MFRL Agent.
#         """
#         with logging_redirect_tqdm():
#             need_evaluate = False
#             need_reset = True
#             for _ in trange(self.maximum_steps):
#                 if need_reset:
#                     epi_reward = 0.0
#                     step_counter = 0
#                     state = self.env.reset()
#                     need_reset = False
#                 action = self.agent.select_action_from_policy(state)
#                 # Do action is for the environment.
#                 next_state, reward, done, _ = self.env.step(action)
#                 # Small action is for training.
#                 self.memory.add(state, action, reward, next_state, done)
#                 state = next_state
#                 step_counter += 1
#                 epi_reward += reward
#                 if len(self.memory) > self.batch_size:
#                     for _ in range(self.G):
#                         self.agent.train_policy(self.memory, batch_size=self.batch_size)
#                     self.counter += 1
#                 if self.counter % self.evaluate_interval == 0:
#                     need_evaluate = True
#                 if done or ((step_counter % self.episode_steps) == 0):
#                     logging.info("Training: " + str(epi_reward))
#                     need_reset = True
#                     if need_evaluate:
#                         self.evaluate()
#                         need_evaluate = False

#     def agent_selection(self, agent_name):
#         """
#         Create an agent
#         :param agent_name:
#         """
#         if agent_name == "SAC":
#             actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
#             self.agent = SAC(actor_network=actor,
#                              critic_network=critic,
#                              action_num=self.action_dim,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              device=torch.device(self.device),
#                              reward_scale=1.0)

#         if agent_name == "TQC":
#             actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = TQC_Critic(observation_size=self.state_dim,
#                                 num_actions=self.action_dim,
#                                 num_quantiles=25,
#                                 num_critics=5)
#             self.agent = TQC(actor_network=actor,
#                              critic_network=critic,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              top_quantiles_to_drop=2,
#                              action_num=self.action_dim,
#                              device=self.device
#                              )

#         if agent_name == "Fully_Expand":
#             actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
#             self.agent = Fully_Expand(self,
#                                       actor_network=actor,
#                                       gamma=0.99,
#                                       tau=0.005,
#                                       action_num=6,
#                                       actor_lr=0.0003,
#                                       alpha_lr=0.0003,
#                                       horizon=10,
#                                       device='cpu')

#         if agent_name == "Hyper_SAC_Critic":
#             actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = Hyper_Double_SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
#             self.agent = SAC(actor_network=actor,
#                              critic_network=critic,
#                              action_num=self.action_dim,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              device=torch.device(self.device),
#                              reward_scale=1.0)

#         if agent_name == "Hyper_TQC_Critic":
#             actor = Actor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = Hyper_TQC_Critic(observation_size=self.state_dim,
#                                       num_actions=self.action_dim,
#                                       num_quantiles=25,
#                                       num_critics=5)
#             self.agent = TQC(actor_network=actor,
#                              critic_network=critic,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              top_quantiles_to_drop=2,
#                              action_num=self.action_dim,
#                              device=self.device)

#         if agent_name == "Hyper_SAC_all":
#             actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = Hyper_Double_SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
#             self.agent = SAC(actor_network=actor,
#                              critic_network=critic,
#                              action_num=self.action_dim,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              device=torch.device(self.device),
#                              reward_scale=1.0)

#         if agent_name == "Hyper_TQC_all":
#             actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = Hyper_TQC_Critic(observation_size=self.state_dim,
#                                       num_actions=self.action_dim,
#                                       num_quantiles=25,
#                                       num_critics=5)
#             self.agent = TQC(actor_network=actor,
#                              critic_network=critic,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              top_quantiles_to_drop=2,
#                              action_num=self.action_dim,
#                              device=self.device)

#         if agent_name == "Hyper_SAC_actor":
#             actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = SAC_Critic(observation_size=self.state_dim, num_actions=self.action_dim)
#             self.agent = SAC(actor_network=actor,
#                              critic_network=critic,
#                              action_num=self.action_dim,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              device=torch.device(self.device),
#                              reward_scale=1.0)

#         if agent_name == "Hyper_TQC_actor":
#             actor = HyperActor(observation_size=self.state_dim, num_actions=self.action_dim)
#             critic = TQC_Critic(observation_size=self.state_dim,
#                                 num_actions=self.action_dim,
#                                 num_quantiles=25,
#                                 num_critics=5)
#             self.agent = TQC(actor_network=actor,
#                              critic_network=critic,
#                              actor_lr=3e-4,
#                              critic_lr=3e-4,
#                              alpha_lr=3e-4,
#                              gamma=0.99,
#                              tau=0.005,
#                              top_quantiles_to_drop=2,
#                              action_num=self.action_dim,
#                              device=self.device)

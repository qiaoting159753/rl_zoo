import numpy as np
import math
import torch
from torch.nn import functional as F
from datetime import datetime
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm


class Trainer:
    """
    A class that responsible for training and evaluating.

    """

    def __init__(self, generate_results, env, agent, memory, device, name,
                 use_mbrl, logger):

        # Should be Goal Conditioned.
        self.logger = logger
        self.use_mbrl = use_mbrl
        self.name = name
        self.generate_results = generate_results
        self.device = device

        self.batch_size = 128
        self.max_epi_steps = 1000
        self.num_eval = 5
        self.train_world_times = 1
        self.train_agent_times = 1

        self.current_step = 0
        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')

        self.evaluation_array = [[], [], [], [], []]
        # self.env = PusherEnv()
        self.env = env
        self.memory = memory
        self.agent = agent

        num_sample = 9
        num_act_dim = 4
        total = 9 * 9 * 9 * 9
        actions = np.zeros((total, num_act_dim))
        action = np.zeros((num_act_dim,))
        acts = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        counter = 0
        for l in range(num_sample):
            action[3] = acts[l]
            for k in range(num_sample):
                action[2] = acts[k]
                for j in range(num_sample):
                    action[1] = acts[j]
                    for i in range(num_sample):
                        action[0] = acts[i]
                        actions[counter] = action
                        counter += 1
        self.actions_tensor = torch.FloatTensor(actions).to(self.device)

    def observe_critic_actor(self, state):
        """

        """
        # Create action samples
        num_act_dim = self.agent.action_dim
        num_sample = 9  # 10 * 10 * 10 * 10 = 100 * 100 = 10000
        total = num_sample ** num_act_dim
        print(num_act_dim)
        state_tensor = torch.FloatTensor(state).to(device=self.device)
        state_tensor = state_tensor.unsqueeze(dim=0)
        multi_state_tensor = torch.repeat_interleave(state_tensor, total, dim=0)

        # For each dimension, there are 721 different distributions.

        # Get all values
        _, _, _, dist = self.agent.actor_net(state_tensor)
        q1, q2 = self.agent.critic_net(multi_state_tensor, self.actions_tensor)

        q_s = q1[0:9]
        # qs to distribution.
        q_s = F.softmax(q_s, dim=0)

        a_s = (dist.log_prob(self.actions_tensor[0:9]))
        a_s = torch.exp(a_s)
        print(a_s)
        a_s = F.softmax(a_s[:, 0], dim=0)
        print(q_s)
        print(a_s)

        # tq1, tq2 = self.agent.target_critic_net(multi_state_tensor, actions_tensor)
        # Convert q back to dist
        # 9 * 9 * 9 = 729 same


    def evaluate(self, observe=False):
        """

        :param observe:
        """
        total_rewards = 0
        reward_error = 0
        dynamic_error = 0
        counter = 1

        for _ in range(self.num_eval):
            state = self.env.reset()
            for _ in range(self.max_epi_steps):
                action = self.agent.select_action_from_policy(state,
                                                              evaluation=True)
                next_state, reward, done, _ = self.env.step(action)
                total_rewards += reward

                # if observe:
                    # state_tensor = torch.FloatTensor(next_state).to(
                    #     device=self.device)
                    # state_tensor = state_tensor.unsqueeze(dim=0)
                    # actions, _, _ = self.agent.actor_net.sample(state_tensor)

                    # self.observe_critic_actor()

                    # q1s, _ = self.agent.critic_net.sample(state_tensor, actions)
                    # # Normalize
                    # temp_min = torch.min(q1s)
                    # temp_max = torch.max(q1s)
                    # temp_scale = temp_max - temp_min
                    # norm_q1s = (q1s - temp_min) / temp_scale

                    # # Reward Prediction.
                    # pred_mean, _ = self.agent.world_model.pred_rewards(
                    #     obs=state_tensor, actions=actions)
                    # pred_mean = pred_mean.item()
                    # reward_error += abs(pred_mean - reward)

                    # # World model prediction
                    # pred_next_state, _, _, _ = self.agent.world_model.pred_next_states(
                    #     obs=state_tensor, actions=actions)
                    # pred_next_state = pred_next_state.detach().cpu().numpy().squeeze()
                    # dynamic_error += (np.mean((pred_next_state - next_state) ** 2))

                    # # uncert 1
                    # total_uncert1 += vi(pred_mean, pred_var)
                    # # uncert 2
                    # total_uncert2 += sampling(pred_mean, pred_var)
                    # # uncert 3
                    # total_uncert3 += mean_std(pred_mean, pred_var)

                counter += 1
                state = next_state
                if done:
                    break
        avg_rewards = total_rewards / self.num_eval
        reward_error /= counter
        dynamic_error /= counter

        self.evaluation_array[0].append(avg_rewards)
        self.evaluation_array[1].append(reward_error)
        self.evaluation_array[2].append(dynamic_error)
        self.evaluation_array[3].append(self.current_step)
        self.evaluation_array[4].append(0)
        eval_array = np.array(self.evaluation_array)

        self.logger.info(f'Evaluation: {total_rewards / self.num_eval}')

        if self.generate_results:
            # Save the metrics
            file_name = (self.env.domain + "_" + self.env.task + "_" +
                         self.name + "_" + self.date_and_time)
            np.savetxt(file_name + "_eval_rewards.csv",
                       eval_array, delimiter=",")
            # Save the actor
            # torch.save(self.agent.actor.state_dict(),
            #            file_name + "_actor_params.pth")

    def train_agent(self):
        """
        Train the agent
        :param max_epi_steps: Maximum number of steps for each episode
        """
        state = self.env.reset()
        for _ in range(self.max_epi_steps):
            # Execute action and add to memory.
            if len(self.memory) < self.batch_size + 1:
                action = np.random.uniform(-1, 1, (self.env.action_num,))
            else:
                action = self.agent.select_action_from_policy(state=state,
                                                              evaluation=False)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward, next_state, done)
            # Training the world model and the agent
            if len(self.memory) > self.batch_size:
                if self.use_mbrl and self.agent.type == "mbrl":
                    statistics = self.memory.get_statistics()
                    self.agent.world_model.set_statistics(statistics)
                    for _ in range(self.train_world_times):
                        transitions = self.memory.sample(
                            batch_size=self.batch_size)
                        self.agent.train_world_model(
                            statistics=statistics,
                            transitions=transitions)

                for _ in range(self.train_agent_times):
                    transitions = self.memory.sample(
                        batch_size=self.batch_size)
                    self.agent.train_policy(transitions)

            # Do evaluation for every 200
            self.current_step += 1
            if done:
                break
            # Move to the next state
            state = next_state

    def train_loop(self):
        """
        The main loop. Call Tranin or evaluation.

        """
        # state = self.env.reset()
        # self.observe_critic_actor(state)
        with logging_redirect_tqdm():
            for i in trange(1200):
                self.evaluate()
                # if i == 50:
                #     state = self.env.reset()
                #     self.observe_critic_actor(state)
                self.train_agent()

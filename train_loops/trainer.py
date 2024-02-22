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

        num_sample = 5
        num_act_dim = 4
        total = 5 * 5 * 5 * 5
        as0 = np.zeros((total, num_act_dim))
        as1 = np.zeros((total, num_act_dim))
        as2 = np.zeros((total, num_act_dim))
        as3 = np.zeros((total, num_act_dim))

        action = np.zeros((num_act_dim,))
        acts = [-0.8, -0.4, 0.0, 0.4, 0.8]

        counter = 0
        for l in range(num_sample):
            action[3] = acts[l]
            for k in range(num_sample):
                action[2] = acts[k]
                for j in range(num_sample):
                    action[1] = acts[j]
                    for i in range(num_sample):
                        action[0] = acts[i]
                        as0[counter] = action
                        counter += 1

        counter = 0
        for l in range(num_sample):
            action[3] = acts[l]
            for k in range(num_sample):
                action[2] = acts[k]
                for j in range(num_sample):
                    action[0] = acts[j]
                    for i in range(num_sample):
                        action[1] = acts[i]
                        as1[counter] = action
                        counter += 1

        counter = 0
        for l in range(num_sample):
            action[3] = acts[l]
            for k in range(num_sample):
                action[1] = acts[k]
                for j in range(num_sample):
                    action[0] = acts[j]
                    for i in range(num_sample):
                        action[2] = acts[i]
                        as2[counter] = action
                        counter += 1

        counter = 0
        for l in range(num_sample):
            action[1] = acts[l]
            for k in range(num_sample):
                action[2] = acts[k]
                for j in range(num_sample):
                    action[0] = acts[j]
                    for i in range(num_sample):
                        action[3] = acts[i]
                        as3[counter] = action
                        counter += 1

        self.as0 = torch.FloatTensor(as0).to(self.device)
        self.as1 = torch.FloatTensor(as1).to(self.device)
        self.as2 = torch.FloatTensor(as2).to(self.device)
        self.as3 = torch.FloatTensor(as3).to(self.device)

    def observe_critic_actor(self, state):
        """

        """
        # Create action samples
        state_tensor = torch.FloatTensor(state).to(device=self.device)
        state_tensor = state_tensor.unsqueeze(dim=0)
        multi_state_tensor = torch.repeat_interleave(state_tensor, 5**4, dim=0)

        # Same states, same action distributions.
        _, _, _, dist = self.agent.actor_net(state_tensor)
        # Same states, different actions.
        q_0, _ = self.agent.critic_net(multi_state_tensor, self.as0)
        q_1, _ = self.agent.critic_net(multi_state_tensor, self.as1)
        # q_2, _ = self.agent.critic_net(multi_state_tensor, self.as2)
        # q_3, _ = self.agent.critic_net(multi_state_tensor, self.as3)

        # Dim 0
        total_kld_0 = 0
        for i in range(125):
            # For first dimension
            q_s_0 = q_0[i*5:i*5+5]
            # qs to distribution.
            q_s_0 = F.softmax(q_s_0, dim=0)
            a_s_0 = (dist.log_prob(self.as0[i*5:i*5+5]))
            a_s_0 = torch.exp(a_s_0)
            a_s_0 = F.softmax(a_s_0[:, 0], dim=0)
            kld0 = F.kl_div(q_s_0, a_s_0)
            total_kld_0 += kld0

        for i in range(125):
            # For first dimension
            q_s_1 = q_1[i*5:i*5+5]
            # qs to distribution.
            q_s_1 = F.softmax(q_s_1, dim=0)
            a_s_1 = (dist.log_prob(self.as1[i*5:i*5+5]))
            a_s_1 = torch.exp(a_s_1)
            a_s_1 = F.softmax(a_s_1[:, 1], dim=0)
            kld1 = F.kl_div(q_s_1, a_s_1)

        # for i in range(125):
        #     # For first dimension
        #     q_s_2 = q_2[i*5:i*5+5]
        #     # qs to distribution.
        #     q_s_2 = F.softmax(q_s_2, dim=0)
        #     a_s_2 = (dist.log_prob(self.as2[i*5:i*5+5]))
        #     a_s_2 = torch.exp(a_s_2)
        #     a_s_2 = F.softmax(a_s_2[:, 2], dim=0)
        #     kld2 = F.kl_div(q_s_2, a_s_2)
        #
        # for i in range(125):
        #     # For first dimension
        #     q_s_3 = q_3[i*5:i*5+5]
        #     # qs to distribution.
        #     q_s_3 = F.softmax(q_s_3, dim=0)
        #     a_s_3 = (dist.log_prob(self.as3[i*5:i*5+5]))
        #     a_s_3 = torch.exp(a_s_3)
        #     a_s_3 = F.softmax(a_s_3[:, 3], dim=0)
        #     kld3 = F.kl_div(q_s_3, a_s_3)

        # Re-shuffle the buffer.
        # for j in range(5):
        #
        #
        #     j = j * 5
        #     q_s_1 = q_1[]
            # print(q_s)
            # print(a_s)

            # For second dimension



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
                # if i == 10:
                #     state = self.env.reset()
                #     self.observe_critic_actor(state)
                self.train_agent()

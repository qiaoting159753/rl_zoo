import math

import numpy as np
from datetime import datetime

import torch
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import torch.nn.functional as F

class Trainer:
    """
    A class that responsible for training and evaluating.

    """

    def __init__(self, env, agent, memory, device, name, logger):

        # Should be Goal Conditioned.
        self.logger = logger
        self.name = name
        self.device = device

        self.batch_size = 256
        self.max_epi_steps = 1000
        self.num_eval = 10

        self.current_step = 0
        self.date_and_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')

        self.evaluation_array = [[], [], [], [], []]

        self.env = env
        self.memory = memory
        self.agent = agent

    def evaluate(self):
        """

        :param observe:
        """
        total_rewards = 0
        reward_errors = 0
        dynamic_errors = 0
        counter = 1

        for _ in range(self.num_eval):
            state = self.env.reset()
            for _ in range(self.max_epi_steps):
                action = self.agent.select_action_from_policy(state, evaluation=True)
                next_state, reward, done, _ = self.env.step(action)

                tensor_state = torch.FloatTensor(state).to(self.device).unsqueeze(dim=0)
                tensor_action = torch.FloatTensor(action).to(self.device).unsqueeze(dim=0)
                tensor_next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(dim=0)

                pred, _, _, _ = self.agent.world_model.pred_next_states(tensor_state, tensor_action)
                model_error = F.mse_loss(pred, tensor_next_state)
                model_error = model_error.item()

                pred_rwd, _, _ = self.agent.world_model.pred_rewards(tensor_state, tensor_action, tensor_next_state)
                pred_rwd = pred_rwd.detach().cpu().item()
                reward_error = math.sqrt((pred_rwd - reward) ** 2)

                reward_errors += reward_error
                dynamic_errors += model_error
                total_rewards += reward
                counter += 1

                state = next_state
                if done:
                    break

        avg_rewards = total_rewards / self.num_eval
        reward_errors /= counter
        dynamic_errors /= counter

        self.evaluation_array[0].append(avg_rewards)
        self.evaluation_array[1].append(dynamic_errors)
        self.evaluation_array[2].append(reward_errors)
        self.evaluation_array[3].append(self.current_step)
        self.evaluation_array[4].append(0.0)
        eval_array = np.array(self.evaluation_array)

        self.logger.info(f'Evaluation: {total_rewards / self.num_eval}')

        if len(self.name) > 5:
            # Save the metrics
            file_name = (self.name + "_" + self.date_and_time)
            np.savetxt(file_name + "_eval_rewards.csv",
                       eval_array, delimiter=",")

    def train_agent(self):
        """
        Train the agent
        :param max_epi_steps: Maximum number of steps for each episode
        """
        state = self.env.reset()
        if len(self.memory) > self.batch_size:
            statistics = self.memory.get_statistics()
            self.agent.world_model.set_statistics(statistics)
        for _ in range(self.max_epi_steps):
            # Execute action and add to memory.
            if len(self.memory) < self.batch_size + 1:
                action = self.env.sample_action()
            else:
                action = self.agent.select_action_from_policy(state=state, evaluation=False)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward, next_state, done)
            # Training the world model and the agent
            if len(self.memory) > self.batch_size:
                if self.agent.type == "mbrl":
                    if len(self.memory) == (self.batch_size + 1):
                        # First time set statics
                        statistics = self.memory.get_statistics()
                        self.agent.world_model.set_statistics(statistics)
                    # Train world model many times.
                    if self.current_step % 5 == 0:
                        self.agent.train_world_model(self.memory, self.batch_size)
                self.agent.train_policy(self.memory, self.batch_size)
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
        with logging_redirect_tqdm():
            for _ in trange(1000):
                self.train_agent()
                self.evaluate()

    # def observe_critic_actor(self, state):
    #     """
    #     For Q evaluation policy vs Q
    #     """
    #     num_sample = 5
    #     num_act_dim = 6
    #     total = 5 * 5 * 5 * 5 * 5 * 5
    #     as0 = np.zeros((total, num_act_dim))
    #     as1 = np.zeros((total, num_act_dim))
    #
    #     action = np.zeros((num_act_dim,))
    #     acts = [-0.8, -0.4, 0.0, 0.4, 0.8]
    #
    #     counter = 0
    #     for l in range(num_sample):
    #         action[5] = acts[l]
    #         for k in range(num_sample):
    #             action[4] = acts[k]
    #             for j in range(num_sample):
    #                 action[3] = acts[j]
    #                 for i in range(num_sample):
    #                     action[2] = acts[i]
    #                     for h in range(num_sample):
    #                         action[1] = acts[h]
    #                         for g in range(num_sample):
    #                             action[0] = acts[g]
    #                             as0[counter] = action
    #                             counter += 1
    #
    #     counter = 0
    #     for l in range(num_sample):
    #         action[5] = acts[l]
    #         for k in range(num_sample):
    #             action[4] = acts[k]
    #             for j in range(num_sample):
    #                 action[3] = acts[j]
    #                 for i in range(num_sample):
    #                     action[2] = acts[i]
    #                     for h in range(num_sample):
    #                         action[1] = acts[h]
    #                         for g in range(num_sample):
    #                             action[0] = acts[g]
    #                             as0[counter] = action
    #                             counter += 1
    #
    #     self.as0 = torch.FloatTensor(as0).to(self.device)
    #     self.as1 = torch.FloatTensor(as1).to(self.device)
    #
    #     # Create action samples
    #     state_tensor = torch.FloatTensor(state).to(device=self.device)
    #     state_tensor = state_tensor.unsqueeze(dim=0)
    #     multi_state_tensor = torch.repeat_interleave(state_tensor, 5 ** 6,
    #                                                  dim=0)
    #
    #     # Same states, same action distributions.
    #     _, _, _, dist = self.agent.actor_net(state_tensor)
    #     # Same states, different actions.
    #     q_0, _ = self.agent.critic_net(multi_state_tensor, self.as0)
    #     # q_2, _ = self.agent.critic_net(multi_state_tensor, self.as2)
    #     # q_3, _ = self.agent.critic_net(multi_state_tensor, self.as3)
    #
    #     # Dim 0
    #     total_kld_0 = 0.0
    #     for i in range(3125):
    #         # For first dimension
    #         q_s_0 = q_0[i * 5:i * 5 + 5]
    #         # qs to distribution.
    #         q_s_0 = F.softmax(q_s_0, dim=0)
    #         a_s_0 = (dist.log_prob(self.as0[i * 5:i * 5 + 5]))
    #         a_s_0 = torch.exp(a_s_0)
    #         a_s_0 = F.softmax(a_s_0[:, 0], dim=0)
    #         kld0 = F.kl_div(q_s_0, a_s_0)
    #         total_kld_0 += kld0
    #     self.logger.info(f"{total_kld_0.item()}")
    #     return total_kld_0.item()
    # state_tensor = torch.FloatTensor(next_state).to(
    #     device=self.device)
    # state_tensor = state_tensor.unsqueeze(dim=0)
    # actions, _, _,_ = self.agent.actor_net.sample(state_tensor)

    # self.observe_critic_actor()

    # q1s, _ = self.agent.critic_net.sample(state_tensor, actions)
    # # Normalize
    # temp_min = torch.min(q1s)
    # temp_max = torch.max(q1s)
    # temp_scale = temp_max - temp_min
    # norm_q1s = (q1s - temp_min) / temp_scale

    # Reward Prediction.
    # pred_mean, _ = self.agent.world_model.pred_rewards(
    #     obs=state_tensor, actions=actions)
    # pred_mean = pred_mean.item()
    # reward_error += abs(pred_mean - reward)

    # World model prediction
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

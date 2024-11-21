"""
Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting."



This code runs automatic entropy tuning
"""
import copy
import logging
import os
import numpy as np
import torch
import math

from rl_zoo.utils import PrioritizedReplayBuffer, denormalize_observation_delta

from rl_zoo.networks.world_models import World_Model
import torch.nn.functional as F


class Immersive_Reweight_Dyna_SAC:
    """
    Use the Soft Actor Critic as the Actor Critic framework.
    """

    def __init__(
            self,
            actor_network: torch.nn.Module,
            critic_network: torch.nn.Module,
            world_network: World_Model,
            gamma: float,
            tau: float,
            action_num: int,
            actor_lr: float,
            critic_lr: float,
            alpha_lr: float,
            num_samples: int,
            horizon: int,
            device: torch.device,
            threshold: float,
    ):
        self.threshold_scale = threshold
        self.type = "mbrl"
        self.device = device

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(self.device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        # Other Variables
        self.gamma = gamma
        self.tau = tau
        # Switches
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_num = action_num

        self.learn_counter = 0
        self.policy_update_freq = 1
        # optimizer
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        # Set to initial alpha to 1.0 according to other baselines.
        self.log_alpha = torch.tensor(np.log(1.0)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_num
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # World model
        self.world_model = world_network
        self.batch_size = None

    @property
    def _alpha(self):
        """
        A variatble decide to what extend entropy shoud be valued.
        """
        return self.log_alpha.exp()

    # pylint: disable-next=unused-argument to keep the same interface
    def select_action_from_policy(
            self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        """
        Select a action for executing. It is the only channel that an agent
        will communicate the the actual environment.
        """
        # note that when evaluating this algorithm we need to select mu as
        # action so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net(state_tensor)
            else:
                (_, _, action) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def _train_policy(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
            weights: torch.Tensor,
    ) -> None:
        if weights is None:
            weights = torch.ones(rewards.shape).to(self.device)
        info = {}
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)
            target_q_one, target_q_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                    torch.minimum(target_q_one, target_q_two) - self._alpha * next_log_pi
            )
            q_target = rewards + self.gamma * (1 - dones) * target_q_values
        assert (len(q_target.shape) == 2) and (q_target.shape[1] == 1)
        q_target = q_target.detach()
        q_values_one, q_values_two = self.critic_net(states, actions)

        # critic_loss_one = F.mse_loss(q_values_one, q_target)
        td_error1 = (q_target - q_values_one)  # * weights
        td_error2 = (q_target - q_values_two)  # * weights
        critic_loss_one = 0.5 * (td_error1.pow(2) * weights).mean()
        critic_loss_two = 0.5 * (td_error2.pow(2) * weights).mean()
        critic_loss_total = critic_loss_one + critic_loss_two
        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()
        ##################     Update the Actor Second     ####################
        pi, first_log_p, _ = self.actor_net(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = ((self._alpha * first_log_p) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(
                self.log_alpha * (first_log_p + self.target_entropy).detach()
        ).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(
                    self.target_critic_net.parameters(), self.critic_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

    def train_world_model(self, memory: PrioritizedReplayBuffer,
                          batch_size: int):
        """
        Sample the buffer again for training the world model can reach higher rewards.

        :param experiences:
        """
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, _, _ = experiences
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        self.world_model.train_world(states, actions, next_states)

        # (
        #     states,
        #     actions,
        #     rewards,
        #     next_states,
        #     _,
        # ) = experiences
        #
        # states = torch.FloatTensor(np.asarray(states)).to(self.device)
        # actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        # rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        # next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        # assert len(states.shape) >= 2
        # assert len(actions.shape) == 2
        # assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        # assert len(next_states.shape) >= 2
        # # # Step 1 train the world model.
        # self.world_model.train_world(
        #     states=states,
        #     actions=actions,
        #     next_states=next_states
        # )
        # self.world_model.train_reward(next_states=next_states, actions=actions, rewards=rewards)

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device).unsqueeze(1)

        # Step 2 train as usual
        self._train_policy(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            weights=torch.ones(rewards.shape).to(self.device),
        )
        # Step 3 Dyna add more statistics
        self._dyna_generate_and_train(next_states=next_states)

    def set_statistics(self, stats: dict) -> None:
        self.world_model.set_statistics(stats)

    def _dyna_generate_and_train(self, next_states: torch.Tensor) -> None:
        """
        Only off-policy Dyna will work.
        :param next_states:
        """
        pred_states = []
        pred_actions = []
        pred_rs = []
        pred_n_states = []
        weights = []

        with torch.no_grad():
            pred_state = next_states
            for _ in range(self.horizon):
                pred_state = torch.repeat_interleave(pred_state, self.num_samples, dim=0)
                # This part is controversial. But random actions is empirically better.
                rand_acts = np.random.uniform(-1, 1, (pred_state.shape[0], self.action_num))
                pred_acts = torch.FloatTensor(rand_acts).to(self.device)
                # [2560, 18]
                pred_next_state, _, norm_means_, norm_vars_ = self.world_model.pred_next_states(
                    pred_state, pred_acts
                )
                pred_reward = self.reward_function(pred_state, pred_acts, pred_next_state)
                uncert = self.sampling(pred_state, norm_means_, norm_vars_)
                # Q, A, R
                weights.append(uncert)
                pred_next_state[:, -2:] = pred_state[:, -2:]
                pred_states.append(pred_state)
                pred_actions.append(pred_acts.detach())
                pred_rs.append(pred_reward.detach())
                pred_n_states.append(pred_next_state.detach())
                pred_state = pred_next_state.detach()
            pred_states = torch.vstack(pred_states)
            pred_actions = torch.vstack(pred_actions)
            pred_rs = torch.vstack(pred_rs)
            pred_n_states = torch.vstack(pred_n_states)
            pred_weights = torch.vstack(weights)
            # Pay attention to here! It is dones in the Cares RL Code!
            pred_dones = torch.FloatTensor(np.zeros(pred_rs.shape)).to(self.device)
            # states, actions, rewards, next_states, not_dones
        print(pred_states.shape)
        print(pred_rs.shape)
        print(pred_n_states.shape)
        print(pred_weights.shape)

        self._train_policy(
            pred_states, pred_actions, pred_rs, pred_n_states, pred_dones, pred_weights
        )

    def reward_function(self, curr_states, actions, next_states):
        target_goal_tensor = curr_states[:, -2:]
        object_current = next_states[:, -4:-2]
        sq_diff = (target_goal_tensor - object_current) ** 2
        # [256, 1]
        goal_distance_after = torch.sqrt(torch.sum(sq_diff, dim=1)).unsqueeze(dim=1)
        pred_reward = torch.round((-goal_distance_after + 70), decimals=2)
        mask1 = goal_distance_after <= 10
        mask2 = goal_distance_after > 70
        pred_reward[mask1] = 800
        pred_reward[mask2] = 0
        return pred_reward

    def sampling(self, curr_states, pred_means, pred_vars):
        """
        High std means low uncertainty. Therefore, divided by 1

        :param pred_means: [num_model, batch_size * 10, observation_dim]
        :param pred_vars:
        :return:
        """

        with torch.no_grad():
            # 5 models. Each predict 10 next_states.
            r_s = []
            act_logs = []
            q_s = []
            # For each model
            for i in range(pred_means.shape[0]):
                sample_times = 10
                samples = torch.distributions.Normal(pred_means[i], pred_vars[i]).sample([sample_times])
                # For each sampling
                for i in range(sample_times):
                    samples[i] = denormalize_observation_delta(samples[i], self.world_model.statistics)
                    samples[i] += curr_states
                    pred_act, log_pi, _ = self.actor_net(samples[i])
                    act_logs.append(log_pi)
                    # pred_rwd1 = self.world_model.pred_rewards(samples[i])
                    rewards = self.reward_function(curr_states, pred_act, samples[i])
                    r_s.append(rewards)
                    qa1, qa2 = self.target_critic_net(samples[i], pred_act)
                    q_a = torch.minimum(qa1, qa2)
                    q_s.append(q_a)
            r_s = torch.stack(r_s)
            act_logs = torch.stack(act_logs)
            q_s = torch.stack(q_s)

            var_r = torch.var(r_s, dim=0)
            var_a = torch.var(act_logs, dim=0)
            var_q = torch.var(q_s, dim=0)

            mean_a = torch.mean(act_logs, dim=0, keepdim=True)
            mean_q = torch.mean(q_s, dim=0, keepdim=True)
            diff_a = act_logs - mean_a
            diff_q = q_s - mean_q
            cov_aq = torch.mean(diff_a * diff_q, dim=0)

            mean_r = torch.mean(r_s, dim=0, keepdim=True)
            diff_r = r_s - mean_r
            cov_rq = torch.mean(diff_r * diff_q, dim=0)
            cov_ra = torch.mean(diff_r * diff_a, dim=0)

            gamma_sq = self.gamma * self.gamma
            total_var = var_r + gamma_sq * var_a + gamma_sq * var_q + gamma_sq * 2 * cov_aq + \
                        gamma_sq * 2 * cov_rq + gamma_sq * 2 * cov_ra
            # # For actor: alpha^2 * var_a + var_q
            min_var = torch.min(total_var)
            max_var = torch.max(total_var)
            # As (max-min) decrease, threshold should go down.
            threshold = self.threshold_scale * (max_var - min_var) + min_var
            total_var[total_var <= threshold] = threshold
            total_var += 0.00000001
            total_stds = 1 / total_var
        return total_stds.detach()

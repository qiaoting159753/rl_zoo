import numpy as np
import torch
from rl_zoo.agents.mbrl import Dyna_SAC_NS
from rl_zoo.networks.world_models import (
    World_Model,
)
from rl_zoo.utils.helpers import denormalize_observation_delta
import logging

class Immerseive_Weighting_Dyna_SAC_NS(Dyna_SAC_NS):
    """
    Dyna of SAC with Next state to predict rewards.

    """
    def __init__(self, actor_network: torch.nn.Module, critic_network: torch.nn.Module, world_network: World_Model,
                 gamma: float, tau: float, action_num: int, actor_lr: float, critic_lr: float, alpha_lr: float,
                 num_samples: int, horizon: int, threshold: float, device: torch.device, train_reward: bool,
                 train_both: bool, gripper: bool):
        super().__init__(actor_network, critic_network, world_network, gamma, tau, action_num, actor_lr, critic_lr,
                         alpha_lr, num_samples, horizon, device, train_reward, train_both, gripper)
        self.threshold = threshold
        logging.info("----------------------------------------------------------------")
        logging.info("---- I am runing the Immersive_Weighting_Dyna_SAC_NS Agent! ----")
        logging.info("----------------------------------------------------------------")

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
        weights = weights.to(self.device)
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
                if self.gripper:
                    pred_reward = self.reward_function(pred_state, pred_acts, pred_next_state)
                    pred_next_state[:, -2:] = pred_state[:, -2:]
                else:
                    pred_reward, _ = self.world_model.pred_rewards(observation=pred_state,
                                                                   action=pred_acts,
                                                                   next_observation=pred_next_state)
                uncert = self.sampling(pred_state, norm_means_, norm_vars_)
                # Q, A, R
                weights.append(uncert)

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
        self._train_policy(
            pred_states, pred_actions, pred_rs, pred_n_states, pred_dones, pred_weights
        )

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
            threshold = self.threshold * (max_var - min_var) + min_var
            total_var[total_var <= threshold] = threshold
            # Inverse variance.
            weights = 1 / total_var
            # Normalization
            new_min_var = torch.min(weights)
            new_max_var = torch.max(weights)
            weights = (weights - new_min_var) / (new_max_var - new_min_var)
            weights += 0.0001
        return weights.detach()


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
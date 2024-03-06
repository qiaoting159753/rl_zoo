"""
A MBRL class that implemented all MBRL algorithms for SAC.
"""
import copy
import numpy as np
import torch
from utils import soft_update


class MBRL_STEVE_CRITIC:
    """
    A MBRL class that implemented all MBRL algorithms for SAC.
    """

    def __init__(
            self,
            actor_network,
            critic_network,
            world_network,
            gamma,
            tau,
            action_num,
            actor_lr,
            critic_lr,
            alpha_lr,
            num_samples,
            horizon,
            device):

        self.type = "mbrl"
        # Switches
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_num = action_num
        self.on_policy = True

        # Other Variables
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.batch_size = None

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        # Set to initial alpha to 1.0 according to other baselines.
        self.log_alpha = torch.tensor(np.log(1.0)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_num

        # optimizer
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr
                                                    )
        # World model
        self.world_model = world_network
        self.world_model.to(device)
        self.learn_counter = 0
        self.policy_update_freq = 1

    @property
    def alpha(self):
        """
        A variatble decide to what extend entropy shoud be valued.
        """
        return self.log_alpha.exp()

    # Only interact with the environment.
    def select_action_from_policy(self, state, evaluation=False, noise_scale=0):
        """
        Select a action for executing. It is the only channel that an agent
        will communicate the the actual environment.

        """
        # note that when evaluating this algorithm we need to select mu as
        # action so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
                self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net.sample(state_tensor)
            else:
                (_, _, action) = self.actor_net.sample(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def true_train_policy(self, states, actions, rewards, next_states, dones):
        """
        Update the critic first, the critic have to learn the correct action.
        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param dones:
        """
        not_dones = 1 - dones
        with torch.no_grad():
            # # For next episodes used
            # pred_all_next_obs = next_states.unsqueeze(dim=0)
            # pred_all_next_rewards = torch.zeros(rewards.shape).unsqueeze(dim=0)
            # q_means = []
            # q_vars = []
            # for hori in range(self.horizon):
            #     horizon_rewards_list = []
            #     horizon_obs_list = []
            #     horizon_q_list = []
            #     for stat in range(pred_all_next_obs.shape[0]):
            #         pred_action, pred_log_pi, _ = self.actor_net.sample(pred_all_next_obs[stat])
            #         pred_q1, pred_q2 = self.target_critic_net(pred_all_next_obs[stat], pred_action)
            #         pred_q3, pred_q4 = self.critic_net(pred_all_next_obs[stat], pred_action)
            #         # V = Q - alpha * logi
            #         pred_v1 = pred_q1 - self.alpha.detach() * pred_log_pi
            #         pred_v2 = pred_q2 - self.alpha.detach() * pred_log_pi
            #         pred_v3 = pred_q3 - self.alpha.detach() * pred_log_pi
            #         pred_v4 = pred_q4 - self.alpha.detach() * pred_log_pi
            #         # Predict a set of reward first
            #         _, pred_rewards = self.world_model.pred_rewards(obs=pred_all_next_obs[stat], actions=pred_action)
            #         temp_disc_rewards = []
            #         # For each predict reward.
            #         for rwd in range(pred_rewards.shape[0]):
            #             disc_pred_reward = not_dones * (self.gamma ** (hori + 1)) * pred_rewards[rwd]
            #             if hori > 0:
            #                 # Horizon = 1, 2, 3, 4, 5
            #                 disc_sum_reward = pred_all_next_rewards[stat] + disc_pred_reward
            #             else:
            #                 disc_sum_reward = not_dones * disc_pred_reward
            #             temp_disc_rewards.append(disc_sum_reward)
            #             assert rewards.shape == not_dones.shape == disc_sum_reward.shape
            #             # Q = r + disc_rewards + pred_v
            #             pred_tq1 = rewards + disc_sum_reward + not_dones * (self.gamma ** (hori + 2)) * pred_v1
            #             pred_tq2 = rewards + disc_sum_reward + not_dones * (self.gamma ** (hori + 2)) * pred_v2
            #             pred_tq3 = rewards + disc_sum_reward + not_dones * (self.gamma ** (hori + 2)) * pred_v3
            #             pred_tq4 = rewards + disc_sum_reward + not_dones * (self.gamma ** (hori + 2)) * pred_v4
            #             horizon_q_list.append(pred_tq1)
            #             horizon_q_list.append(pred_tq2)
            #             horizon_q_list.append(pred_tq3)
            #             horizon_q_list.append(pred_tq4)
            #         # Observation Level
            #         if hori < (self.horizon - 1):
            #             _, pred_obs, _, _ = self.world_model.pred_next_states(pred_all_next_obs[stat], pred_action)
            #             horizon_obs_list.append(pred_obs)
            #             horizon_rewards_list.append(torch.stack(temp_disc_rewards))
            #     # Horizon level.
            #     if hori < (self.horizon - 1):
            #         pred_all_next_obs = torch.vstack(horizon_obs_list)
            #         pred_all_next_rewards = torch.vstack(horizon_rewards_list)
            #     #     # Statistics of target q
            #     h_0 = torch.stack(horizon_q_list)
            #     mean_0 = torch.mean(h_0, dim=0)
            #     q_means.append(mean_0)
            #     var_0 = torch.var(h_0, dim=0)
            #     var_0[torch.abs(var_0) < 0.001] = 0.001
            #     var_0 = 1.0 / var_0
            #     q_vars.append(var_0)
            # all_means = torch.stack(q_means)
            # all_vars = torch.stack(q_vars)
            # total_vars = torch.sum(all_vars, dim=0)
            # for n in range(self.horizon):
            #     all_vars[n] /= total_vars
            # target_q = torch.sum(all_vars * all_means, dim=0)
            next_actions, next_log_pi, _ = self.actor_net.sample(next_states)
            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            target_q_values = (torch.minimum(target_q_values_one, target_q_values_two) - self.alpha * next_log_pi)
            target_q = rewards + self.gamma * (1 - dones) * target_q_values

        assert (len(target_q.shape) == 2) and (target_q.shape[1] == 1)
        target_q = target_q.detach()
        current_q1, current_q2 = self.critic_net(states, actions)
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
        # loss_1, loss_2 = self.critic_net.loss(obs, actions, target_q)
        critic1_loss = 0.5 * (td_error1.pow(2)).mean()
        critic2_loss = 0.5 * (td_error2.pow(2)).mean()
        critic_loss = critic1_loss + critic2_loss
        # Optimize the critic
        self.critic_net_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_net_optimiser.step()

        # Delay the use of this Q network
        p_i, first_log_p, _ = self.actor_net.sample(next_states)
        qf1_pi, qf2_pi = self.critic_net(next_states, p_i)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * first_log_p) - min_qf_pi).mean()
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # optimize the temperature alpha.
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (first_log_p + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(
                self.target_critic_net.parameters(), self.critic_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

    def train_world_model(self, experiences):
        """
        Sample the buffer again for training the world model can reach higher rewards.

        :param experiences:
        """
        (
            states,
            actions,
            rewards,
            next_states,
            _,
            next_actions,
            next_rewards,
        ) = experiences
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(
            self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(
            self.device)
        next_rewards = (
            torch.FloatTensor(np.asarray(next_rewards)).to(
                self.device).unsqueeze(1)
        )
        next_actions = torch.FloatTensor(np.asarray(next_actions)).to(
            self.device)
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_rewards.shape) == 2 and next_rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        # # Step 1 train the world model.
        self.world_model.train_world(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_actions=next_actions,
            next_rewards=next_rewards,
        )

    def train_policy(self, experiences):
        """
        Interface to training loop.

        """
        self.learn_counter += 1
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = experiences
        self.batch_size = len(states)
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device).unsqueeze(1)
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        # Step 2 train as usual
        self.true_train_policy(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

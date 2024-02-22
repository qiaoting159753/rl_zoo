"""
A MBRL class that implemented all MBRL algorithms for SAC.
"""
import copy
import numpy as np
import torch
from utils import soft_update
import logging

class MBRL_DYNA_MNM_SAC:
    """
    A MBRL class that implemented all MBRL algorithms for SAC.
    """

    def __init__(self, actor_network, critic_network, world_model, gamma, tau,
                 state_dim, action_dim, actor_lr, critic_lr, alpha_lr, horizon,
                 sample_times, on_policy, use_bound, device):

        super().__init__()
        self.type = "mbrl"
        # Switches
        self.on_policy = on_policy
        self.sample_times = sample_times
        self.horizon = horizon
        self.use_bounded_active = use_bound

        # Other Variables
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.log_alpha = torch.tensor(np.log(1.0)).float().to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(),
                                                 lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        # World Model
        self.world_model = world_model
        self.world_model.to(device)

    @property
    def alpha(self):
        """
        Control updating speed between rewards and other networks.
        :return:
        """
        return self.log_alpha.exp()

    # Only interact with the environment.
    def select_action_from_policy(self, state, evaluation=False,
                                  noise_scale=0):
        """
        Make decisions with the trained policy.
        # note that when evaluating this algorithm we need to select mu as action
        # _, _, action = self.actor_net.sample(state_tensor)

        :param obs:
        :param sample:
        :return:
        """
        assert len(state.shape) == 1
        self.actor_net.eval()
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(
            dim=0)
        # Evaluation
        if evaluation:
            _, _, action, _ = self.actor_net.sample(state_tensor)
        # Exploration
        else:
            action, _, _, _ = self.actor_net.sample(state_tensor)
        assert action.ndim == 2 and action.shape[0] == 1
        action = action.detach()
        action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def update_critic(self, obs, actions, rewards, next_obs, not_dones):
        """
        Update the critic first, the critic have to learn the correct action.
        :param obs:
        :param actions:
        :param rewards:
        :param next_obs:
        :param not_dones:
        """
        with torch.no_grad():
            next_actions, next_log_pi, _, _ = self.actor_net.sample(next_obs)
            q_1, q_2 = self.target_critic_net(next_obs, next_actions)
            t_q = torch.minimum(q_1, q_2) - self.alpha * next_log_pi
            target_q = rewards + self.gamma * not_dones * t_q
        target_q = target_q.detach()
        assert (len(target_q.shape) == 2) and (target_q.shape[1] == 1)

        current_q1, current_q2 = self.critic_net(obs, actions)
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
        # loss_1, loss_2 = self.critic_net.loss(obs, actions, target_q)
        critic1_loss = 0.5 * (td_error1.pow(2)).mean()
        critic2_loss = 0.5 * (td_error2.pow(2)).mean()
        critic_loss = critic1_loss + critic2_loss
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs):
        """

        Update the actor with respect to critics.
        :param obs:
        """
        # MFRL
        action, first_log_pi, _, _ = self.actor_net.sample(obs)
        actor_q1, actor_q2 = self.critic_net(obs, action)
        actor_q = torch.min(actor_q1, actor_q2)
        # Q - alpha * log = V
        actor_loss = -(actor_q - self.alpha.detach() * first_log_pi).mean()
        # optimize the actor.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # optimize the temperature alpha.
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (
                first_log_pi + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def train_with_true(self, transitions):
        """
        Train with the transitions.

        :param transitions:
        """
        states, actions, rewards, next_states, not_dones, _, _ = transitions
        # Update current Q network
        self.update_critic(states, actions, rewards, next_states, not_dones)
        # Update Actor
        self.update_actor_and_alpha(states)
        # Update target Q network
        soft_update(self.critic_net, self.target_critic_net, self.tau)

    def train_policy(self, transitions):
        """
        Train with true transition and then dyna generated transitions.
        :param transitions:
        """
        # Train with normal samples.
        states, actions, rewards, next_states, not_dones, _, _ = transitions

        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        assert len(not_dones.shape) == 2 and not_dones.shape[1] == 1

        self.train_with_true(transitions)
        self.dyna_generate_and_train(next_states)

    def train_world_model(self, statistics, transitions):
        """

        :param statistics:
        :param transitions:
        """
        self.world_model.set_statistics(statistics)
        states, actions, rewards, next_states, _, next_actions, next_rewards = transitions
        # mask the nones and zeros out.
        ok_masks = []
        for i in range(len(states)):
            if torch.sum(next_actions[i]) == 0 or next_rewards[i] == np.inf:
                ok_masks.append(False)
            else:
                ok_masks.append(True)
        states = states[ok_masks]
        actions = actions[ok_masks]
        rewards = rewards[ok_masks]
        next_states = next_states[ok_masks]
        next_actions = next_actions[ok_masks]
        next_rewards = next_rewards[ok_masks]
        self.world_model.train_world(states, actions, rewards, next_states,
                                     next_actions, next_rewards)

    def dyna_generate_and_train(self, next_states):

        """
        Only off-policy Dyna will work.
        :param next_states:
        """
        pred_states = []
        pred_actions = []
        pred_rewards = []
        pred_next_states = []

        pred_state = next_states
        for _ in range(self.horizon):
            ###    Rewards   ###
            pred_state = torch.repeat_interleave(pred_state,
                                                 self.sample_times, dim=0)
            # if True:
            pred_acts, _, _, _ = self.actor_net.sample(pred_state)
            # else:
            #     random_actions = []
            #     for _ in range(pred_state.shape[0]):
            #         for _ in range(self.sample_times):
            #             pred_act = np.random.uniform(-1, 1,
            #                                          (self.action_dim,))
            #             random_actions.append(pred_act)
            #     pred_acts = torch.FloatTensor(np.array(random_actions)).to(
            #         self.device)
            ###    Predictions   ###
            pred_next_state, _, _, _ = self.world_model.pred_next_states(
                pred_state, pred_acts)
            pred_reward_temp, _ = self.world_model.pred_rewards(pred_state, pred_acts)
            pred_reward_temp = pred_reward_temp.detach()
            pred_reward_temp[pred_reward_temp <=0.01] = 0.01
            pred_reward_temp[pred_reward_temp >=0.99] = 0.99

            # Uncertainty measures.
            scores = self.world_model.discriminator(pred_next_state)
            scores = scores.detach()
            scores[scores <=0.01] = 0.01
            scores[scores >=0.99] = 0.99

            pred_reward = torch.log(scores/(1-scores)) + torch.log(pred_reward_temp.detach())

            ###    Append    ###
            pred_states.append(pred_state)
            pred_actions.append(pred_acts.detach())
            pred_rewards.append(pred_reward)
            pred_next_states.append(pred_next_state.detach())
            ###    Move on to the next    ###
            pred_state = pred_next_state.detach()
        pred_states = torch.vstack(pred_states)
        pred_actions = torch.vstack(pred_actions)
        pred_rewards = torch.vstack(pred_rewards)
        pred_next_states = torch.vstack(pred_next_states)
        pred_not_dones = torch.FloatTensor(np.ones(pred_rewards.shape)).to(
            self.device)
        # states, actions, rewards, next_states, not_dones
        self.train_with_true((pred_states, pred_actions, pred_rewards,
                              pred_next_states, pred_not_dones, None, None))

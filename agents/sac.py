"""
A MBRL class that implemented all MBRL algorithms for SAC.
"""
import copy
import numpy as np
import torch
from utils import soft_update


class SAC:
    """
    A MBRL class that implemented all MBRL algorithms for SAC.
    """

    def __init__(self, actor_network, critic_network, gamma, tau,
                 state_dim, action_dim, actor_lr, critic_lr, alpha_lr, device):

        super().__init__()
        self.type = "mfrl"

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

        :param evaluation:
        :param state:
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
            _, _, action, _ = self.actor_net.forward(state_tensor)
        # Exploration
        else:
            action, _, _, _ = self.actor_net.forward(state_tensor)
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
            # The world model error penalty term should be added here.
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
        # Can generate more times.No TD structure. No world model needed?
        # Still can expand the Q value? How to?
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

    def train_with_true(self, states, actions, rewards, next_states, not_dones):
        """
        Train with the transitions.

        :param transitions:
        """
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

        self.train_with_true(states, actions, rewards, next_states, not_dones)

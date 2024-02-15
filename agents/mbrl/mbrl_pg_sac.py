"""
A MBRL class that implemented all MBRL algorithms for SAC.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from utils import soft_update
from utils import vi
from torch.autograd import Variable
from .classifier import Generator, Discriminator


class MBRL_PG_SAC:
    """
    A MBRL class that implemented all MBRL algorithms for SAC.
    """

    def __init__(self, actor_network, critic_network, world_model, gamma, tau,
                 state_dim, action_dim, actor_lr, critic_lr, alpha_lr, horizon,
                 use_bound, device):

        super().__init__()
        self.batch_size = None
        self.type = "mbrl"
        # Switches
        self.use_normal = False
        self.sample_times = 512
        self.horizon = horizon
        self.use_bounded_active = use_bound
        # self.dyna_use_uncertainty = use_dyna

        # Other Variables
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor Critic.
        self.generator = Generator(latent_variable=1,
                                   observation_size=self.state_dim,
                                   num_actions=self.action_dim)

        self.discriminator = Discriminator(observation_size=self.state_dim,
                                           num_actions=self.action_dim)

        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(),
                                               lr=0.001)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(),
                                               lr=0.001)

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
            _, _, action = self.actor_net.forward(state_tensor)
        # Exploration
        else:
            action, _, _ = self.actor_net.forward(state_tensor)
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
            next_actions, next_log_pi, _ = self.actor_net.sample(next_obs)
            q_1, q_2 = self.target_critic_net.sample(next_obs, next_actions)
            t_q = torch.minimum(q_1, q_2) - self.alpha * next_log_pi
            target_q = rewards + self.gamma * not_dones * t_q
        target_q = target_q.detach()
        assert (len(target_q.shape) == 2) and (target_q.shape[1] == 1)

        current_q1, current_q2 = self.critic_net.forward(obs, actions)
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
        action, first_log_pi, _ = self.actor_net.sample(obs)
        actor_q1, actor_q2 = self.critic_net.sample(obs, action)
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

    def train_policy(self, transitions):
        """

        :param transitions:
        """
        # Train with normal samples.
        states, actions, rewards, next_states, not_dones, _, _ = transitions

        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        assert len(not_dones.shape) == 2 and not_dones.shape[1] == 1

        # Update current Q network
        self.update_critic(states, actions, rewards, next_states, not_dones)
        # Update Actor
        self.update_actor_and_alpha(states)
        # Update target Q network
        soft_update(self.critic_net, self.target_critic_net, self.tau)

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

"""
Original Paper: https://arxiv.org/abs/1812.05905
Code based on: https://github.com/SamsungLabs/tqc_pytorch

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch

from utils import soft_update_params, quantile_huber_loss_f
from utils.memory import PrioritizedReplayBuffer


class MVE_TQC:
    """

    """
    def __init__(
            self,
            env,
            actor_network: torch.nn.Module,
            critic_network: torch.nn.Module,
            gamma: float,
            tau: float,
            top_quantiles_to_drop: int,
            action_num: int,
            actor_lr: float,
            critic_lr: float,
            alpha_lr: float,
            device: str,
    ):
        self.steve = False
        self.type = "policy"
        self.env = env
        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        self.horizon = 10
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.gamma = gamma
        self.tau = tau
        self.top_quantiles_to_drop = top_quantiles_to_drop

        self.quantiles_total = (
                self.critic_net.num_quantiles * self.critic_net.num_critics
        )
        self.learn_counter = 0
        self.policy_update_freq = 1
        self.device = device
        self.target_entropy = -action_num
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )
        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    # pylint: disable-next=unused-argument
    def select_action_from_policy(
            self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select tanh(mean) as action
        # so _, _, action = self.actor_net(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (
                    action,
                    _,
                    _,
                ) = self.actor_net(state_tensor)
            else:
                (
                    _,
                    _,
                    action,
                ) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp()

    def _update_critics(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            cumulative_rewards = rewards
            pred_s = next_states
            tree_mask = dones.squeeze().bool()
            for hori in range(self.horizon):
                # As normal
                pred_a, _, _ = self.actor_net(pred_s)
                # Pred the future
                pred_s, pred_r, pred_done = self.env.tensor_query(pred_s, pred_a)
                pred_s = pred_s.to(self.device)
                pred_r = pred_r.to(self.device)
                pred_done = pred_done.bool().to(self.device)
                # Before adding pred to mask
                pred_r[tree_mask, :] = 0.0
                cumulative_rewards += pred_r * (self.gamma ** (hori + 1))
                # Kill the branch with the previous
                tree_mask = torch.logical_or(tree_mask, pred_done.squeeze())
            q_target = cumulative_rewards
        q_values = self.critic_net(states, actions)
        critic_loss_total = quantile_huber_loss_f(q_values, q_target)
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

    def _fake_critic(self, actions, states):
        next_states, rewards, dones = self.env.tensor_query(states, actions)
        cumulative_rewards = rewards
        tree_mask = dones.bool().squeeze().to(self.device)
        pred_s = next_states
        for hori in range(self.horizon):
            # As normal
            pred_a, _, _ = self.actor_net(pred_s)
            # Pred the future
            pred_s, pred_r, pred_done = self.env.tensor_query(pred_s, pred_a)
            pred_s = pred_s.to(self.device)
            pred_r = pred_r.to(self.device)
            pred_done = pred_done.bool().squeeze().to(self.device)
            # Before adding pred to mask
            pred_r[tree_mask, :] = 0.0
            cumulative_rewards += pred_r * (self.gamma ** (hori + 1))
            # Kill the branch with the previous
            tree_mask = torch.logical_or(tree_mask, pred_done.squeeze())
        return cumulative_rewards

    def _update_actor(self, states):
        new_action, log_pi, _ = self.actor_net(states)
        # mean_qf_pi = self.critic_net(states, new_action).mean(2).mean(1, keepdim=True)
        # mean_qf_pi = self._fake_critic(actions=new_action, states=states)
        actor_loss = (self.alpha * log_pi - mean_qf_pi).mean()
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        # update the temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        self.learn_counter += 1
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences
        batch_size = len(states)
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)
        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)
        # # Update the Critics
        # self._update_critics(states, actions, rewards, next_states, dones)
        # if self.learn_counter % self.policy_update_freq == 0:
        #     soft_update_params(self.critic_net, self.target_critic_net, self.tau)
        # Update the Actor
        self._update_actor(states)

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)
        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")

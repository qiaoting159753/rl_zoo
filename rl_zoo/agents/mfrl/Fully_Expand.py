"""
Original Paper: https://arxiv.org/abs/1812.05905
Code based on: https://github.com/SamsungLabs/tqc_pytorch

This code runs automatic entropy tuning
"""

import logging
import os

import numpy as np
import torch

from rl_zoo.utils import PrioritizedReplayBuffer


class Fully_Expand:
    """
    Replace the critic with a fake critic built by forward kinematcis world model.

    """
    def __init__(
            self,
            env,
            actor_network: torch.nn.Module,
            gamma: float,
            tau: float,
            action_num: int,
            actor_lr: float,
            alpha_lr: float,
            horizon: int,
            device: str,
    ):
        self.steve = False
        self.type = "policy"
        self.env = env
        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        self.horizon = horizon
        self.gamma = gamma
        self.tau = tau

        self.learn_counter = 0
        self.policy_update_freq = 1
        self.device = device
        self.target_entropy = -action_num
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
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
        mean_qf_pi = self._fake_critic(actions=new_action, states=states)
        actor_loss = (self.alpha * log_pi - mean_qf_pi).mean()
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        # update the temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1
        experiences = memory.sample_uniform(batch_size)
        states, _, _, _, _, _ = experiences
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        self._update_actor(states)

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)
        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        logging.info("models has been loaded...")

from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import copy
from torch import nn
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, denormalize_observation_delta
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
from agents.networks.world_models.deterministic.prior_network_utils import NormalInvGamma, evidential_regression


class Prior_World_Model(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r,
                 hidden_size,
                 device):
        self.statistics = None
        self.device = device
        self.observation_size = observation_size

        self.world_model = nn.Sequential(
            nn.Linear(self.observation_size + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            NormalInvGamma(hidden_size, self.observation_size),
        )

        self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r)

        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)
        self.weight_set_samples = []

        self.reward_optimizers = torch.optim.Adam(self.reward_model.parameters(), lr=l_r)
        self.reward_model.to(self.device)
        self.world_model.to(self.device)

    def set_statistics(self, statistics: dict) -> None:
        """
        Update all statistics for normalization for all world models and the
        ensemble itself.

        :param (Dictionary) statistics:
        """
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                statistics[key] = torch.FloatTensor(statistics[key]).to(self.device)
        self.statistics = statistics
        self.world_model.statistics = statistics

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train for On-policy flush training use.
        :param states:
        :param actions:
        :param next_states:
        """
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        s_n_a = torch.cat((states, actions), dim=1)
        pred = self.world_model.forward(s_n_a)
        loss = evidential_regression(pred, delta_targets_normalized, lamb=1e-2)
        self.world_optimizers.zero_grad()
        loss.backward()
        self.world_optimizers.step()

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_n_a = torch.cat((observation, actions), dim=1)
        pred = self.world_model(s_n_a)
        mu, v, alpha, beta = (d.squeeze() for d in pred)
        prediction = denormalize_observation_delta(mu.unsqueeze(dim=0), self.statistics)
        prediction += observation
        return prediction, None, None, None

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        s_n_a = torch.cat((observation, actions), dim=1)
        with torch.no_grad():
            pred = self.world_model(s_n_a)
        mu, v, alpha, beta = (d.squeeze() for d in pred)
        var = torch.sqrt(beta / (v * (alpha - 1)))
        uncert = torch.mean(var).item()
        return uncert, 0.0

    def pred_rewards(self, observation: torch.Tensor,
                     action: torch.Tensor, next_observation: torch.Tensor):
        """
        predict reward based on current observation and action and next state
        """
        pred_reward, reward_var = self.reward_model.forward(observation, action, next_observation)
        return pred_reward, None, reward_var

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        """
        Train the reward with S, A, SN to eliminate difference between them.
        Args:
            states:
            actions:
            next_states:
            rewards:
        """
        self.reward_optimizers.zero_grad()
        rwd_mean, rwd_var = self.reward_model.forward(states, actions, next_states)
        # reward_loss = F.mse_loss(rwd_mean, sub_rewards)
        reward_loss = F.gaussian_nll_loss(input=rwd_mean, target=rewards, var=rwd_var).mean()
        reward_loss.backward()
        self.reward_optimizers.step()

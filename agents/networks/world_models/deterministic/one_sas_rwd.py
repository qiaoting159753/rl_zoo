import logging
import math
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim

from agents.networks.world_models.deterministic import (
    Probabilistic_Dynamics,
)
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
from agents.networks.world_models import World_Model

from utils.helpers import normalize_observation_delta
from utils import denormalize_observation_delta


class One_Dyna_One_SAS_Reward(World_Model):
    """
    This class consist of an ensemble of all components for critic update.
    Q_label = REWARD + gamma * (1 - DONES) * Q(NEXT_STATES).
    """

    def __init__(self, observation_size: int, num_actions: int, l_r: float, device: str,
                 hidden_size: int = 128):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.device = device
        self.world_model = Probabilistic_Dynamics(observation_size=observation_size, num_actions=num_actions,
                                                  hidden_size=hidden_size)
        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)
        self.world_optimizers = optim.Adam(self.world_model.parameters(), lr=l_r)
        self.reward_optimizers = optim.Adam(self.reward_model.parameters(), lr=l_r)
        self.reward_model.to(self.device)
        self.world_model.to(self.device)
        self.statistics = {}

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

    def pred_rewards(self, observation: torch.Tensor,
                     action: torch.Tensor, next_observation: torch.Tensor):
        """
        predict reward based on current observation and action and next state
        """
        pred_reward, reward_var = self.reward_model.forward(observation, action, next_observation)
        return pred_reward, None, reward_var

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        """
        Predict the next state based on the current state and action.

        The output is
        Args:
            observation:
            actions:

        Returns:
            prediction: Single prediction, probably mean.
            all_predictions: all means from different model.
            predictions_norm_means: normalized means.
            predictions_vars: normalized vars.
        """
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        # Predict delta
        prediction, n_mean, n_var = self.world_model.forward(observation, actions)
        if torch.any(torch.isnan(prediction)):
            logging.info("Predicting all Nans")
            sys.exit()
        prediction += observation
        return prediction, None, n_mean, n_var

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train the world with S, A, SN. Different sub-batch.

        Args:
            states:
            actions:
            next_states:
        """
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert (
                states.shape[1] + actions.shape[1]
                == self.num_actions + self.observation_size
        )
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        _, n_mean, n_var = self.world_model.forward(states, actions)
        model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
        self.world_optimizers.zero_grad()
        model_loss.backward()
        self.world_optimizers.step()

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

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        """
        Estimate uncertainty.

        :param observation:
        :param actions:
        """
        sample_times = 10
        _, _, mean, var = self.pred_next_states(observation, actions)
        dyna_uncert = torch.sum(var.squeeze()).item()
        # Sample next state several times, and estimate reward uncertianty.
        sample1 = torch.distributions.Normal(mean, var).sample([sample_times])
        sample1 = sample1.squeeze()
        sample1i = denormalize_observation_delta(sample1, self.world_model.statistics)
        sample1i += observation
        multi_observation = torch.repeat_interleave(observation, sample_times, dim=0)
        multi_reward = torch.repeat_interleave(actions, sample_times, dim=0)
        reward, _, rwd_var = self.pred_rewards(multi_observation, multi_reward, sample1i)
        sample2 = torch.distributions.Normal(reward, rwd_var).sample([sample_times])
        sample2 = torch.reshape(sample2, (sample_times ** 2,))
        rwd_uncert = torch.var(sample2).item()
        return dyna_uncert, rwd_uncert

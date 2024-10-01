from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import copy

from agents.networks.world_models import World_Model

from utils.helpers import normalize_observation_delta, denormalize_observation_delta

from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
from utils.common import MLP

from agents.networks.world_models.bayesian.bayesian_javirantoran_optim import SGLD, pSGLD


class Bayesian_World_Model_SGLD_JA(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r,
                 hidden_size,
                 device):
        self.statistics = None
        self.device = device
        self.observation_size = observation_size

        self.world_model = MLP(input_size=(observation_size + num_actions), output_size=2 * observation_size,
                               hidden_sizes=[128, 128, 128])

        self.world_optimizers = pSGLD(self.world_model.parameters(), lr=l_r)

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

    def save_sampled_net(self, max_samples):
        """
        Sample the network parameters with optimizer?
        :param max_samples:
        :return:
        """
        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)
        self.weight_set_samples.append(copy.deepcopy(self.world_model.state_dict()))
        return None

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
        pred_s = self.world_model.forward(s_n_a)
        n_mean_delta = pred_s[:, :self.observation_size:]
        n_log_delta = pred_s[:, self.observation_size:]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        model_loss = F.gaussian_nll_loss(input=n_mean_delta, target=delta_targets_normalized, var=normalized_var).mean()
        self.world_optimizers.zero_grad()
        model_loss.backward()
        self.world_optimizers.step()

        self.save_sampled_net(max_samples=20)

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_n_a = torch.cat((observation, actions), dim=1)
        pred_s = self.world_model.forward(s_n_a)
        n_mean_delta = pred_s[:, :self.observation_size]
        prediction = denormalize_observation_delta(n_mean_delta, self.statistics)
        prediction += observation
        return prediction, None, None, None

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        length = len(self.weight_set_samples)
        uncert = 0.0
        if  length > 0:
            s_n_a = torch.cat((observation, actions), dim=1)
            Nsamples = len(self.weight_set_samples)
            sample_times = 10

            predictions = []
            # iterate over all saved weight configuration samples
            for idx, weight_dict in enumerate(self.weight_set_samples):
                if idx == Nsamples:
                    break
                self.world_model.load_state_dict(weight_dict)
                out_temp = self.world_model(s_n_a)
                n_mean_delta = out_temp[:, :self.observation_size:]
                n_log_delta = out_temp[:, self.observation_size:]
                logvar = torch.tanh(n_log_delta)
                normalized_var = torch.exp(logvar)
                sample1 = torch.distributions.Normal(n_mean_delta, normalized_var).sample([sample_times])
                sample1 = sample1.squeeze()
                predictions.append(sample1)

            predictions = torch.stack(predictions)
            predictions = torch.reshape(predictions, (length * sample_times, self.observation_size))
            uncert = torch.mean(torch.var(predictions, dim=0)).item()
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

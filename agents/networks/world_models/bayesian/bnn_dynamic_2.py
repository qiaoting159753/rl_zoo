from .bayesian_javirantoran_bbb_lr import BayesLinear_local_reparam
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
from utils import (normalize_observation, denormalize_observation_delta)
import numpy as np


class Bayesian_Dynamics(nn.Module):
    def __init__(self, observation_size, num_actions, hidden_size):
        super(Bayesian_Dynamics,self).__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions

        self.prior_sig = 0.1

        self.layer1 = BayesLinear_local_reparam(observation_size + num_actions, hidden_size, self.prior_sig)
        self.layer2 = BayesLinear_local_reparam(hidden_size, hidden_size, self.prior_sig)
        self.mean_logvar_layer = BayesLinear_local_reparam(hidden_size, 2 * observation_size, self.prior_sig)

        self.statistics = {}

    def forward(self, obs, actions, sample=False):
        assert (obs.shape[1] + actions.shape[1] == self.observation_size +
                self.num_actions)
        # Always normalized obs
        normalized_obs = normalize_observation(obs, self.statistics)
        x = torch.cat((normalized_obs, actions), dim=1)

        tlqw = 0
        tlpw = 0

        # -----------------
        x, lqw, lpw = self.layer1(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = F.relu(x)
        # -----------------
        x, lqw, lpw = self.layer2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = F.relu(x)
        # -----------------
        x, lqw, lpw = self.mean_logvar_layer(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw

        normalized_mean = x[:, :self.observation_size]
        logvar = x[:, self.observation_size:]

        logvar = torch.tanh(logvar)
        normalized_var = torch.exp(logvar)
        # Always denormalized delta
        mean_deltas = denormalize_observation_delta(normalized_mean, self.statistics)
        return mean_deltas, normalized_mean, normalized_var, x, tlqw, tlpw

    def sample_predict(self, obs, actions, N_samples):
        """

        :param obs:
        :param actions:
        :param N_samples:
        :return:
        """
        # Just copies type from x, initializes new vector
        x = torch.cat((obs, actions), dim=1)
        predictions = x.data.new(N_samples, obs.shape[0], self.output_dim)

        tlqw_vec = np.zeros(N_samples)
        tlpw_vec = np.zeros(N_samples)

        for i in range(N_samples):
            _, _, _, y, tlqw, tlpw = self.forward(obs, actions, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw

        return predictions, tlqw_vec, tlpw_vec

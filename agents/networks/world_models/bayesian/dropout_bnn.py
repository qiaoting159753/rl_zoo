import torch
import math
from torch import nn
import numpy as np
import torch.utils
import torch.nn.functional as F
from torch import optim

from utils.nf_util import (
    weight_init,
    normalize_observation,
    normalize_observation_delta,
    denormalize_observation_delta,
)


class Dropout_Dynamics(nn.Module):

    def __init__(self, observation_size, num_actions, hidden_size):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.layer1 = nn.Linear(observation_size + num_actions, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, observation_size)
        self.apply(weight_init)
        self.dropout = nn.Dropout(p=0.3)
        self.dyna_optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.statistics = {}

    def forward(self, obs, actions):
        assert (
                obs.shape[1] + actions.shape[1] == self.observation_size + self.num_actions
        )
        # Always normalized obs
        normalized_obs = normalize_observation(obs, self.statistics)
        x = torch.cat((normalized_obs, actions), dim=1)
        x = self.dropout(self.layer1(x))
        x = F.relu(x)
        x = self.dropout(self.layer2(x))
        x = F.relu(x)
        normalized_mean = self.layer3(x)
        # Always denormalized delta
        mean_deltas = denormalize_observation_delta(normalized_mean, self.statistics)
        return mean_deltas, normalized_mean

    def set_statistics(self, statistics):
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                statistics[key] = torch.FloatTensor(statistics[key]).to(self.device)
        self.statistics = statistics

    def train_world(self, states, actions, next_states):
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        _, normalized_mean = self.forward(states, actions)
        model_loss = F.mse_loss(input=normalized_mean, target=delta_targets_normalized)
        self.dyna_optimizer.zero_grad()
        model_loss.backward()
        self.dyna_optimizer.step()

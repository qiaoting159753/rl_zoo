import torchbnn as bnn
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
from utils.nf_util import (normalize_obs,
                           unnormalize_obs_deltas)


class BNN_Dynamics(nn.Module):
    def __init__(self, observation_size, num_actions, hidden_size):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.layer1 = bnn.BayesLinear(observation_size + num_actions, hidden_size)
        self.layer2 = bnn.BayesLinear(hidden_size, hidden_size)
        self.mean_layer = bnn.BayesLinear(hidden_size, observation_size)
        self.logvar_layer = bnn.BayesLinear(hidden_size, observation_size)

        self.statistics = {}

    def forward(self, obs, actions):
        assert (obs.shape[1] + actions.shape[1] == self.observation_size +
                self.num_actions)
        # Always normalized obs
        normalized_obs = normalize_obs(obs, self.statistics)
        x = torch.cat((normalized_obs, actions), dim=1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        normalized_mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        logvar = torch.tanh(logvar)
        normalized_var = torch.exp(logvar)
        # Always denormalized delta
        mean_deltas = unnormalize_obs_deltas(normalized_mean, self.statistics)
        return mean_deltas, normalized_mean, normalized_var

    def train_world(self):

    def pred_next_states(self, obs, actions):
        self.forward(obs, actions)
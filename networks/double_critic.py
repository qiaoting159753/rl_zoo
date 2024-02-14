import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
from utils import weight_init


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden_dim = 256
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.outputs = {}
        self.apply(weight_init)

    def forward(self, obs, action):
        """

        :param obs:
        :param action:
        :return:
        """
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)
        x_1 = F.relu(self.linear1(obs_action))
        x_1 = F.relu(self.linear2(x_1))
        x_1 = self.linear3(x_1)
        x_2 = F.relu(self.linear4(obs_action))
        x_2 = F.relu(self.linear5(x_2))
        x_2 = self.linear6(x_2)
        self.outputs['q1'] = x_1
        self.outputs['q2'] = x_2
        return x_1, x_2

    def sample(self, obs, act):
        """
        Sample == Forward in this case.
        :param obs:
        :param act:
        :return:
        """
        return self.forward(obs, act)

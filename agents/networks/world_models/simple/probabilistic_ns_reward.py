import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.helpers import weight_init


class Probabilistic_NS_Reward(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, hidden_size: int, normalize:bool):
        """
        Note, This reward function is limited to 0 ~ 1 for dm_control.
        A reward model with fully connected layers. It takes current states (s)
        and current actions (a), and predict rewards (r).
        """
        super().__init__()
        self.normalize = normalize
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(observation_size + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.apply(weight_init)

    def forward(
            self,
            next_observation: torch.Tensor,
            actions: torch.Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward the inputs throught the network.
        Note: For DMCS environment, the reward is from 0~1.
        """
        x = torch.cat((next_observation, actions), dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        rwd_mean = self.linear3(x)
        var_mean = self.linear4(x)
        logvar = torch.tanh(var_mean)
        normalized_var = torch.exp(logvar)
        if self.normalize:
            rwd_mean = F.sigmoid(rwd_mean)
        return rwd_mean, normalized_var

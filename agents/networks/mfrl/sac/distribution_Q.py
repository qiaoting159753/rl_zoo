import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
from utils import weight_init


class DoubleDistributionalQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden_dim = 256
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear3_1 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.linear6_1 = nn.Linear(hidden_dim, 1)

        self.outputs = dict()
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
        mean_x1 = self.linear3(x_1)

        var_x1 = self.linear3_1(x_1)
        var_x1 = torch.tanh(var_x1)
        var_x1 = torch.exp(var_x1)

        x_2 = F.relu(self.linear4(obs_action))
        x_2 = F.relu(self.linear5(x_2))
        mean_x2 = self.linear6(x_2)

        var_x2 = self.linear6_1(x_2)
        var_x2 = torch.tanh(var_x2)
        var_x2 = torch.exp(var_x2)

        return mean_x1, mean_x2, var_x1, var_x2

    def sample(self, obs, acts):
        """

        :param obs:
        :param acts:
        :return:
        """
        mean_x1, mean_x2, var_x1, var_x2 = self.forward(obs, acts)
        q_1 = torch.distributions.Normal(loc=mean_x1, scale=var_x1)
        q_2 = torch.distributions.Normal(loc=mean_x2, scale=var_x2)
        return q_1.rsample(), q_2.rsample()

    def loss(self, obs, acts, t_1):
        """
        Compute the loss for both network.

        :param obs:
        :param acts:
        :param t_1:
        :return:
        """
        m_1, m_2, v_1, v_2 = self.forward(obs, acts)

        model_loss1 = F.gaussian_nll_loss(input=m_1,
                                          target=t_1,
                                          var=v_1).mean()

        model_loss2 = F.gaussian_nll_loss(input=m_2,
                                          target=t_1,
                                          var=v_2).mean()

        return model_loss1, model_loss2

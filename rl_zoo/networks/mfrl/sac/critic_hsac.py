import torch
from torch import nn
import torch.nn.functional as F


class Hyper_SAC_Critic(nn.Module):
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 hidden_size: list[int] = None,
                 ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]
        self.hidden_size = hidden_size

        # Q1 architecture
        # pylint: disable-next=invalid-name
        # 9 * 64, 64 * 1
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.obs_act_size = self.observation_size + self.num_actions

        self.first_layer = 64

        # Input: 9, Output: 64 * (9 + 1)
        self.Q1_1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.obs_act_size * self.first_layer + self.first_layer),
            nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.first_layer ** 2 + self.first_layer),
            nn.Tanh()
        )

        self.Q1_2 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.first_layer + 1),
            nn.Tanh()
        )

    def forward(self, obs_action):
        q1_wb_1 = self.Q1_1(obs_action)
        q1_w_1 = q1_wb_1[:, :self.obs_act_size * self.first_layer]
        q1_b_1 = q1_wb_1[:, self.obs_act_size * self.first_layer:].unsqueeze(dim=2)
        q1_w_1 = torch.unflatten(q1_w_1, dim=1, sizes=(self.first_layer, self.obs_act_size))
        x_1 = obs_action.unsqueeze(dim=2)
        x_1 = torch.matmul(q1_w_1, x_1) + q1_b_1
        x_1 = F.relu(x_1)

        q1_wb = self.Q1(obs_action)
        q1_w = q1_wb[:, :self.first_layer ** 2]
        q1_b = q1_wb[:, self.first_layer ** 2:].unsqueeze(dim=2)
        q1_w = torch.unflatten(q1_w, dim=1, sizes=(self.first_layer, self.first_layer))
        x_1 = torch.matmul(q1_w, x_1) + q1_b
        x_1 = F.relu(x_1)

        q1_wb_2 = self.Q1_2(obs_action)
        q1_w_2 = q1_wb_2[:, :self.first_layer]
        q1_b_2 = q1_wb_2[:, self.first_layer:].unsqueeze(dim=2)
        q1_w_2 = torch.unflatten(q1_w_2, dim=1, sizes=(1, self.first_layer))
        x_1 = torch.matmul(q1_w_2, x_1) + q1_b_2
        x_1 = x_1.squeeze(dim=2)

        return x_1


class Hyper_Double_SAC_Critic(nn.Module):
    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            hidden_size: list[int] = None,
    ):
        super().__init__()
        self.q1 = Hyper_SAC_Critic(observation_size, num_actions, hidden_size)
        self.q2 = Hyper_SAC_Critic(observation_size, num_actions, hidden_size)

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        x_1 = self.q1(obs_action)
        x_2 = self.q2(obs_action)
        return x_1, x_2

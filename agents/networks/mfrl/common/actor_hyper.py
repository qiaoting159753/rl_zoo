import torch
from torch import nn
from utils import SquashedNormal
from utils.common import HyperMLP


class HyperActor(nn.Module):
    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            hidden_size: list[int] = None,
            log_std_bounds: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]
        if log_std_bounds is None:
            log_std_bounds = [-20, 2]

        self.hidden_size = hidden_size
        self.log_std_bounds = log_std_bounds
        self.hypermlp = HyperMLP(input_size=observation_size, output_size=num_actions*2)

    def forward(self, states):
        result = self.hypermlp(states)
        mu = result[:, :int(result.shape[1]/2)]
        log_std = result[:, int(result.shape[1]/2):]
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)
        return sample, log_pi, dist.mean


class HActor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
        log_std_bounds: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]
        if log_std_bounds is None:
            log_std_bounds = [-20, 2]

        self.hidden_size = hidden_size
        self.log_std_bounds = log_std_bounds

        # Assume the first layer size is 128
        # obs * 64, 64 * action * 2
        self.num_actions = num_actions
        self.observation_size = observation_size

        self.first_size = 64
        # input * 64 + 64
        self.l1_a = nn.Linear(observation_size, 128)
        self.l1_b = nn.Linear(128, (observation_size * self.first_size + self.first_size) )

        # 64 * action + action
        self.l2_a = nn.Linear(observation_size, 128)
        self.l2_b = nn.Linear(128, (self.first_size * num_actions + num_actions))

        # 64 * action + action
        self.l3_a = nn.Linear(observation_size, 128)
        self.l3_b = nn.Linear(128, (self.first_size * num_actions + num_actions))


    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        l1_a = self.l1_a(state)
        l1_wb = self.l1_b(l1_a)
        l1_w = l1_wb[:, :int(self.observation_size * self.first_size)]
        l1_w = torch.unflatten(l1_w, dim=1, sizes=(self.first_size, self.observation_size))
        l1_b = l1_wb[:, int(self.observation_size * self.first_size):].unsqueeze(dim=2)
        state = state.unsqueeze(dim=2)
        x = torch.matmul(l1_w, state) + l1_b

        state = state.squeeze(dim=2)
        l2_a = self.l2_a(state)
        l2_wb = self.l2_b(l2_a)
        l2_w = l2_wb[:, :int(self.first_size * self.num_actions)]
        l2_w = torch.unflatten(l2_w, dim=1, sizes=(self.num_actions, self.first_size))
        l2_b = l2_wb[:, int(self.first_size * self.num_actions):].unsqueeze(dim=2)
        mu = torch.matmul(l2_w, x) + l2_b

        l3_a = self.l3_a(state)
        l3_wb = self.l3_b(l3_a)
        l3_w = l3_wb[:, :int(self.first_size * self.num_actions)]
        l3_w = torch.unflatten(l3_w, dim=1, sizes=(self.num_actions, self.first_size))
        l3_b = l3_wb[:, int(self.first_size * self.num_actions):].unsqueeze(dim=2)
        log_std = torch.matmul(l3_w, x) + l3_b

        mu = mu.squeeze(dim=2)
        log_std = log_std.squeeze(dim=2)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.mean
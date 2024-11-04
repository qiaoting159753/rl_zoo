import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F

class HyperMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.obs_act_size = input_size
        self.first_layer = 64
        self.output_size = output_size

        # Input: 9, Output: 64 * (9 + 1)
        self.Q1_1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_act_size * self.first_layer + self.first_layer),
        )

        self.Q1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.first_layer ** 2 + self.first_layer),
        )

        self.Q1_2 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.first_layer * output_size + output_size),
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
        q1_w_2 = q1_wb_2[:, :self.first_layer * self.output_size]
        q1_b_2 = q1_wb_2[:, self.first_layer * self.output_size:].unsqueeze(dim=2)
        q1_w_2 = torch.unflatten(q1_w_2, dim=1, sizes=(self.output_size, self.first_layer))
        x_1 = torch.matmul(q1_w_2, x_1) + q1_b_2
        x_1 = x_1.squeeze(dim=2)
        return x_1


# Standard Multilayer Perceptron (MLP) network
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()

        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = nn.Linear(input_size, next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size

        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, state):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        return output


# CNN from Nature paper: https://www.nature.com/articles/nature14236
class NatureCNN(nn.Module):
    def __init__(self, observation_size: tuple[int]):
        super().__init__()

        self.cnn_modules = [
            nn.Conv2d(observation_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ]

        self.nature_cnn = nn.Sequential(*self.cnn_modules)

        with torch.no_grad():
            dummy_image = torch.zeros([1, *observation_size])
            n_flatten = self.nature_cnn(torch.FloatTensor(dummy_image))

        self.cnn_modules.append(nn.Linear(n_flatten.shape[1], 512))

        self.nature_cnn = nn.Sequential(*self.cnn_modules)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.nature_cnn(state)
        return output


# Stable version of the Tanh transform - overriden to avoid NaN values through atanh in pytorch
class StableTanhTransform(TanhTransform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)


# These methods are not required for the purposes of SAC and are thus intentionally ignored
# pylint: disable=abstract-method
class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)

        transforms = [StableTanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=False)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
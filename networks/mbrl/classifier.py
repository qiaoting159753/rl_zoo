import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()
        self.linear1 = nn.Linear(observation_size*2+num_actions, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        """
        Forward
        :param state:
        :return:
        """
        x_1 = self.linear1(state)
        x_1 = F.leaky_relu(x_1, negative_slope=0.2, inplace=True)
        x_1 = self.linear2(x_1)
        x_1 = F.leaky_relu(x_1, negative_slope=0.2, inplace=True)
        x_1 = self.linear3(x_1)
        x_1 = F.sigmoid(x_1)
        return x_1


class Generator(nn.Module):
    def __init__(self, latent_variable, observation_size, num_actions):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(latent_variable, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, observation_size*2+num_actions)

    def forward(self, z):
        """

        :param z:
        :return:
        """
        x_1 = self.linear1(z)
        x_1 = F.leaky_relu(x_1, negative_slope=0.2, inplace=True)
        x_1 = self.linear2(x_1)
        x_1 = F.leaky_relu(x_1, negative_slope=0.2, inplace=True)
        x_1 = self.linear3(x_1)
        x_1 = F.tanh(x_1)
        return x_1



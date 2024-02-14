import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Discriminator(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(observation_dim + action_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        """
        Forward
        :param state:
        :return:
        """
        x = self.linear1(state)
        nn.LeakyReLU(0.2, inplace=True)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.linear3(x)
        return x

class Generator(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(observation_size + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


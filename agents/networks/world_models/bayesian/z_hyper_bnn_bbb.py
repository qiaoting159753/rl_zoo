import torch
import torch.nn as nn
import numpy as np
from agents.networks.world_models import World_Model
from torch.autograd import Variable
from utils import normalize_observation_delta
import torch.nn.functional as F
from torch import optim
from agents.networks.world_models.simple import (
    Probabilistic_SAS_Reward,
)
from utils import normalize_observation, denormalize_observation_delta
from utils import MLP


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5 * (weights - self.mu) ** 2 / self.sigma ** 2
        log_coeff = -0.5 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))
        return (exponent + log_coeff).sum()


class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        self.weight_mus_mlp = MLP(self.input_dim, [32, 32, 32], self.input_dim * self.output_dim)
        self.weight_rhos_mlp = MLP(self.input_dim, [32, 32, 32], self.input_dim * self.output_dim)
        self.bias_mus_mlp = MLP(self.input_dim, [32, 32, 32], self.output_dim)
        self.bias_rhos_mlp = MLP(self.input_dim, [32, 32, 32], self.output_dim)

        # self.weight_mus = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        # self.weight_rhos = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -3))
        # self.bias_mus = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        # self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -3))

    def forward(self, x):

        # x: [256 , 23]
        # weight_sample: [23 * 128] -> 256 * 128

        # mus: 256 * 23 * 128
        # rhos : 256 * 23 * 128
        # sampled weights: 256 * 23 * 128

        # [256, 23] [23, 128 ] -> [256, 128]

        batch_size = x.shape[0]
        self.weight_mus = self.weight_mus_mlp(x).view((batch_size, self.input_dim, self.output_dim))
        self.weight_rhos = self.weight_rhos_mlp(x).view((batch_size, self.input_dim, self.output_dim))
        self.bias_mus = self.bias_mus_mlp(x).view((batch_size, self.output_dim))
        self.bias_rhos = self.bias_rhos_mlp(x).view((batch_size, self.output_dim))

        x = x.unsqueeze(dim=1)

        # sample gaussian noise for each weight and each bias
        weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
        bias_epsilons = Variable(self.bias_mus.data.new(self.bias_mus.size()).normal_())
        # calculate the weight and bias stds from the rho parameters
        weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
        bias_stds = torch.log(1 + torch.exp(self.bias_rhos))
        # calculate samples from the posterior from the sampled noise and mus/stds
        weight_sample = self.weight_mus + weight_epsilons * weight_stds
        bias_sample = self.bias_mus + bias_epsilons * bias_stds
        output = torch.matmul(x, weight_sample).squeeze() + bias_sample
        # computing the KL loss term
        prior_cov, varpost_cov = self.prior.sigma ** 2, weight_stds ** 2
        KL_loss = 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * weight_stds.numel()
        KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        KL_loss = KL_loss + 0.5 * ((self.weight_mus - self.prior.mu) ** 2 / prior_cov).sum()
        prior_cov, varpost_cov = self.prior.sigma ** 2, bias_stds ** 2
        KL_loss = KL_loss + 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * bias_stds.numel()
        KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        KL_loss = KL_loss + 0.5 * ((self.bias_mus - self.prior.mu) ** 2 / prior_cov).sum()
        return output, KL_loss, 0.0


class Hyper_BBP_Heteroscedastic_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(Hyper_BBP_Heteroscedastic_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, num_units, gaussian(0, 1))
        self.layer2 = BayesLinear_Normalq(num_units, num_units, gaussian(0, 1))
        self.layer3 = BayesLinear_Normalq(num_units, output_dim, gaussian(0, 1))
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)
        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)
        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)
        x, KL_loss = self.layer3(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)
        return x, KL_loss_total, 0.0

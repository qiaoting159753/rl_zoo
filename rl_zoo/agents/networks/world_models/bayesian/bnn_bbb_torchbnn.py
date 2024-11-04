import torch.nn as nn
import math
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class BayesLinear(Module):
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self):
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias:
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self):
        self.weight_eps = None
        if self.bias:
            self.bias_eps = None

    def forward(self, input, sample=True):
        r"""
        Overriden.
        """
        if sample:
            if self.weight_eps is None:
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
            else:
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps

            if self.bias:
                if self.bias_eps is None:
                    bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
                else:
                    bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu,
                                                                                              self.prior_sigma,
                                                                                              self.in_features,
                                                                                              self.out_features,
                                                                                              self.bias is not None)


class CustomizedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, sigma: float):
        super().__init__()
        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = BayesLinear(prior_mu=0.0,
                                                prior_sigma=sigma,
                                                in_features=input_size,
                                                out_features=next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size
        self.output_layer = BayesLinear(prior_mu=0.0,
                                        prior_sigma=sigma,
                                        in_features=input_size,
                                        out_features=output_size)

    def forward(self, state, sample=True):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state, sample))
        output = self.output_layer(state)
        return output

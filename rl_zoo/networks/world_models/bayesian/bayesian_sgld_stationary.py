import copy
from itertools import compress
import numpy as np
import time
from torch.optim import Optimizer
from collections.abc import MutableSequence
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


class CustomizedMLP(nn.Module):
    def __init__(self,
                 observation_size,
                 device,
                 num_actions,
                 hidden_sizes=None,
                 name=None):
        super().__init__()
        self.name = name
        self.device = device
        self.dataloader = None
        self.observation_size = observation_size
        input_size = observation_size + num_actions
        output_size = observation_size * 2

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )
        self.log_std = nn.Parameter(torch.FloatTensor([-1]))
        self.data = None
        self.target = None

    def reset_parameters(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.log_std.data = torch.FloatTensor([3.])

    def sample(self):
        self.reset_parameters()

    def forward(self, x_x):
        output = self.model(x_x)
        means = output[:, :self.observation_size]
        var_s = output[:, self.observation_size:]
        var_s = torch.tanh(var_s)
        var_s = torch.exp(var_s)
        return means, var_s

    def log_prob(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        self.model.to(self.device)
        mu, var_s = self.forward(data)
        mse = F.mse_loss(mu, target)
        # log_prob = torch.distributions.Normal(mu, F.softplus(self.log_std)).log_prob(self.target).mean()
        log_prob = -F.gaussian_nll_loss(mu, target=target, var=var_s)
        return {'log_prob': log_prob, 'MSE': mse.detach_()}


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class MetropolisHastingsAcceptance():
    def __init__(self):
        pass
    def __call__(self, log_prob_proposal, log_prob_state):
        if not torch.isnan(log_prob_proposal) or not torch.isinf(log_prob_proposal):
            log_ratio = (log_prob_proposal - log_prob_state)
            log_ratio = torch.min(log_ratio, torch.zeros_like(log_ratio))
            log_u = torch.zeros_like(log_ratio).uniform_(0, 1).log()
            log_accept = torch.gt(log_ratio, log_u)
            log_accept = log_accept.bool().item()
            return log_accept, log_ratio
        elif torch.isnan(log_prob_proposal) or torch.isinf(log_prob_proposal):
            exit(f'log_prob_proposal is nan or inf {log_prob_proposal}')
            return False, torch.Tensor([-1])


class SDE_Acceptance():
    def __init__(self):
        pass

    def __call__(self, log_prob_proposal, log_prob_state):
        return True, torch.Tensor([0.])

class Sampler:
    def __init__(self, probmodel, step_size, num_steps, num_chains, burn_in, pretrain, tune):
        self.probmodel = probmodel
        self.chain = None
        self.num_chains = num_chains
        self.step_size = step_size
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.pretrain = pretrain
        self.tune = tune
        test_log_prob = self.probmodel.log_prob(*next(self.probmodel.dataloader.__iter__()))
        assert type(test_log_prob) == dict
        assert list(test_log_prob.keys())[0] == 'log_prob'

    def sample_chains(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def multiprocessing_test(self, wait_time):
        time.sleep(wait_time)
        print(f'Done after {wait_time=} seconds')
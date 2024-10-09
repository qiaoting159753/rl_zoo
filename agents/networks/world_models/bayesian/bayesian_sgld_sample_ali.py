from joblib import Parallel, delayed
import copy
from collections.abc import MutableSequence
from itertools import compress
import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


class CustomizedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.iiiiinput_size = input_size
        self.ooooutput_size = output_size

        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = nn.Linear(input_size, next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size
        self.output_layer = nn.Linear(input_size, output_size)
        self.obs_length = int(output_size / 2)
        self.log_std = nn.Parameter(torch.FloatTensor([-1]))

        self.current_input = torch.zeros((1, self.iiiiinput_size))
        self.current_output = torch.zeros((1, self.ooooutput_size))

    def forward(self, state):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        mean_out = output[:, :self.obs_length]
        var_out = output[:, self.obs_length:]
        logvar = torch.tanh(var_out)
        normalized_var = torch.exp(logvar)
        # output = torch.concatenate((mean_out, normalized_var), dim=1)
        return mean_out, normalized_var

    def reset_parameters(self):
        for module in self.fully_connected_layers:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.log_std.data = torch.FloatTensor([3.])

    def log_prob(self):
        # Loss function?
        mu, var_s = self.forward(self.current_input)
        # log_prob = torch.distributions.Normal(mu, F.softplus(self.log_std)).log_prob(target).mean()
        log_prob = F.gaussian_nll_loss(mu, self.current_output, var_s)
        mse = F.mse_loss(mu, self.current_output)
        return {'log_prob': log_prob, 'MSE': mse.detach_()}


class SDE_Acceptance():
    def __init__(self):
        pass

    def __call__(self, log_prob_proposal, log_prob_state):
        return True, torch.Tensor([0.])


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


class SGLD_Optim(Optimizer):
    def __init__(self, model, step_size=0.001, prior_std=1., addnoise=True):
        weight_decay = 1 / (prior_std ** 2) if prior_std != 0 else 0
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if step_size < 0.0:
            raise ValueError("Invalid learning rate: {}".format(step_size))
        defaults = dict(step_size=step_size, weight_decay=weight_decay, addnoise=addnoise)
        self.model = model
        params = self.model.parameters()
        Optimizer.__init__(self, params=params, defaults=defaults)

    def step(self):
        log_prob = None
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(alpha=weight_decay, other=p.data)
                if group['addnoise']:
                    noise = torch.randn_like(p.data).mul_(group['step_size'] ** 0.5)
                    p.data.add_(grad, alpha=-0.5 * group['step_size'])
                    p.data.add_(noise)
                    if torch.isnan(p.data).any(): exit('Nan param')
                    if torch.isinf(p.data).any(): exit('inf param')
                else:
                    p.data.add_(other=0.5 * grad, alpha=-group['step_size'], )
        return log_prob


class Chain(MutableSequence):
    def __init__(self, probmodel=None):
        super().__init__()
        if probmodel is None:
            self.state_dicts = []
            self.log_probs = []
            self.accepts = []
        if probmodel is not None:
            self.state_dicts = [copy.deepcopy(probmodel.state_dict())]
            log_prob = probmodel.log_prob()
            log_prob['log_prob'].detach_()
            self.log_probs = [copy.deepcopy(log_prob)]
            self.accepts = [True]
            self.last_accepted_idx = 0
            self.running_avgs = {}
            for key, value in log_prob.items():
                self.running_avgs.update({key: RunningAverageMeter(0.99)})
        self.running_accepts = RunningAverageMeter(0.999)

    def __len__(self):
        return len(self.state_dicts)

    def __iter__(self):
        return zip(self.state_dicts, self.log_probs, self.accepts)

    def __delitem__(self):
        raise NotImplementedError

    def __setitem__(self):
        raise NotImplementedError

    def insert(self):
        raise NotImplementedError

    def __repr__(self):
        return f'MCMC Chain: Length:{len(self)} Accept:{self.accept_ratio:.2f}'

    def __getitem__(self, i):
        chain = copy.deepcopy(self)
        chain.state_dicts = self.samples[i]
        chain.log_probs = self.log_probs[i]
        if isinstance(self.accepts[i], bool):
            chain.accepts = [self.accepts[i]]
        return chain

    def __add__(self, other):
        if type(other) in [tuple, list]:
            assert len(
                other) == 3, f"Invalid number of information pieces passed: {len(other)} vs len(Iterable(model, log_prob, accept, ratio))==4"
            self.append(*other)
        elif isinstance(other, Chain):
            self.cat(other)
        return self

    def __iadd__(self, other):
        if type(other) in [tuple, list]:
            assert len(
                other) == 3, f"Invalid number of information pieces passed: {len(other)} vs len(Iterable(model, log_prob, accept, ratio))==4"
            self.append(*other)
        elif isinstance(other, Chain):
            self.cat_chains(other)
        return self

    @property
    def state_idx(self):
        if not hasattr(self, 'state_idx'):
            self.last_accepted_idx = np.where(self.accepts == True)[0][-1]
            return self.last_accepted_idx
        else:
            last_accepted_sample_ = np.where(self.accepts == True)[0][-1]
            assert last_accepted_sample_ == self.last_accepted_idx
            assert self.accepts[self.last_accepted_idx] == True
            return self.last_accepted_idx

    @property
    def samples(self):
        return list(compress(self.state_dicts, self.accepts))

    @property
    def accept_ratio(self):
        return sum(self.accepts) / len(self.accepts)

    @property
    def state(self):
        return {'state_dict': self.state_dicts[self.last_accepted_idx],
                'log_prob': self.log_probs[self.last_accepted_idx]}

    def cat_chains(self, other):
        assert isinstance(other, Chain)
        self.state_dicts += other.state_dicts
        self.log_probs += other.log_probs
        self.accepts += other.accepts
        for key, value in other.running_avgs.items():
            self.running_avgs[key].avg = 0.5 * self.running_avgs[key].avg + 0.5 * other.running_avgs[key].avg

    def append(self, probmodel, log_prob, accept):
        params_state_dict = copy.deepcopy(probmodel)
        assert isinstance(log_prob, dict)
        assert type(log_prob['log_prob']) == torch.Tensor
        assert log_prob['log_prob'].numel() == 1
        log_prob['log_prob'].detach_()

        self.accepts.append(accept)
        self.running_accepts.update(1 * accept)
        if accept:
            self.state_dicts.append(params_state_dict)
            self.log_probs.append(copy.deepcopy(log_prob))
            self.last_accepted_idx = len(self.state_dicts) - 1
            for key, value in log_prob.items():
                self.running_avgs[key].update(value.item())
        elif not accept:
            self.state_dicts.append(False)
            self.log_probs.append(False)


class SGLD_Chain:
    def __init__(self, probmodel, step_size, num_steps, burn_in):
        self.probmodel = probmodel
        self.chain = Chain(probmodel=self.probmodel)
        self.step_size = step_size
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.optim = SGLD_Optim(self.probmodel, step_size=step_size)
        self.acceptance = SDE_Acceptance()

    def propose(self):
        self.optim.zero_grad()
        log_prob = self.probmodel.log_prob()
        (log_prob['log_prob']).backward()
        self.optim.step()
        return log_prob, self.probmodel

    def sample_chain(self):
        self.probmodel.reset_parameters()
        self.chain = Chain(probmodel=self.probmodel)
        for step in range(self.num_steps):
            proposal_log_prob, sample = self.propose()
            accept, log_ratio = self.acceptance(proposal_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
            self.chain += (self.probmodel, proposal_log_prob, accept)
            if not accept:
                if torch.isnan(proposal_log_prob['log_prob']):
                    print(self.chain.state)
                    exit()
                self.probmodel.load_state_dict(self.chain.state['state_dict'])
        self.chain = self.chain[self.burn_in:]
        return self.chain


class SGLD_Sampler:
    def __init__(self, probmodel, step_size=0.001, num_steps=100, num_chains=3, burn_in=10):
        self.parallel_chains = None
        self.chain = None
        self.probmodel = probmodel
        self.step_size = step_size
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.num_chains = num_chains
        
    def sample_chains(self):
        if self.num_chains > 1:
            self.parallel_chains = [SGLD_Chain(copy.deepcopy(self.probmodel),
                                               step_size=self.step_size,
                                               num_steps=self.num_steps,
                                               burn_in=self.burn_in)
                                    for _ in range(self.num_chains)]
            chains = Parallel(n_jobs=self.num_chains)(delayed(chain.sample_chain)() for chain in self.parallel_chains)
        elif self.num_chains == 1:
            chain = SGLD_Chain(copy.deepcopy(self.probmodel),
                               step_size=self.step_size,
                               num_steps=self.num_steps,
                               burn_in=self.burn_in)
            chains = [chain.sample_chain()]
        self.chain = Chain(probmodel=self.probmodel)
        for chain in chains:
            self.chain += chain
        return chains

    def __str__(self):
        return 'SGLD'

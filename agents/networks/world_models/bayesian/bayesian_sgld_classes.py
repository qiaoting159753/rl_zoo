import copy
from itertools import compress
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from collections.abc import MutableSequence
from joblib import Parallel, delayed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


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


class SDE_Acceptance():
    def __init__(self):
        pass

    def __call__(self, log_prob_proposal, log_prob_state):
        return True, torch.Tensor([0.])


class SGLD_Optim(Optimizer):
    def __init__(self, model, step_size=0.1, prior_std=1., addnoise=True):
        weight_decay = 1 / (prior_std ** 2) if prior_std != 0 else 0
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if step_size < 0.0:
            raise ValueError("Invalid learning rate: {}".format(step_size))
        defaults = dict(step_size=step_size, weight_decay=weight_decay, addnoise=addnoise)
        self.model = model
        params = self.model.parameters()
        Optimizer.__init__(self, params=params, defaults=defaults)
        self.tune_params = {'delta': 0.65,
                            't0': 10,
                            'gamma': .05,
                            'kappa': .75,
                            'mu': 0.,
                            'H': 0,
                            'log_eps': 1.}

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

    def tune(self, accepts):
        avg_acc = sum(accepts) / len(accepts)
        if avg_acc < 0.001:
            scale = 0.1
        elif avg_acc < 0.05:
            scale = 0.5
        elif avg_acc < 0.20:
            # PyMC: 0.9
            scale = 0.5
        elif avg_acc > 0.99:
            scale = 10.
        elif avg_acc > 0.75:
            scale = 2.
        elif avg_acc > 0.5:
            # PyMC: 1.1
            scale = 1.1
        else:
            scale = 0.9
        for group in self.param_groups:
            group['step_size'] *= scale

    def dual_average_tune(self, accepts, t, alpha):
        assert 0 < alpha <= 1., f'{alpha=}'
        delta, t0, gamma, kappa, mu, H, log_eps = self.tune_params.values()
        H = (1 - 1 / (t + t0)) * H + 1 / (t + t0) * (delta - alpha)
        log_eps_t = mu - t ** 0.5 / gamma * H
        log_eps = t ** (-kappa) * log_eps_t + (1 - t ** (-kappa)) * log_eps
        self.tune_params["H"] = H
        self.tune_params["log_eps"] = log_eps
        for group in self.param_groups:
            group["step_size"] = np.exp(log_eps)


class Chain(MutableSequence):
    def __init__(self, probmodel=None):
        super().__init__()
        if probmodel is None:
            self.state_dicts = []
            self.log_probs = []
            self.accepts = []

        if probmodel is not None:
            self.state_dicts = [copy.deepcopy(probmodel.state_dict())]
            log_prob = probmodel.log_prob(*next(probmodel.dataloader.__iter__()))
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

    def __repr__(self):
        return f'MCMC Chain: Length:{len(self)} Accept:{self.accept_ratio:.2f}'

    def __delitem__(self):
        raise NotImplementedError

    def __setitem__(self):
        raise NotImplementedError

    def insert(self):
        raise NotImplementedError

    def __getitem__(self, i):
        chain = copy.deepcopy(self)
        chain.state_dicts = self.samples[i]
        chain.log_probs = self.log_probs[i]
        chain.accepts = self.accepts[i]
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
        params_state_dict = copy.deepcopy(probmodel.state_dict())
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
    def __init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune):
        self.probmodel = probmodel
        self.chain = Chain(probmodel=self.probmodel)
        self.step_size = step_size
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.pretrain = pretrain
        self.tune = tune
        self.optim = SGLD_Optim(self.probmodel, step_size=step_size)
        self.acceptance = SDE_Acceptance()

    @torch.enable_grad()
    def propose(self):
        self.optim.zero_grad()
        log_prob = self.probmodel.log_prob(*next(self.probmodel.dataloader.__iter__()))
        (-log_prob['log_prob']).backward()
        self.optim.step()
        return log_prob, self.probmodel

    def __repr__(self):
        return 'SGLD'

    def tune_step_size(self):
        tune_interval_length = 100
        print(f'Tuning: Init Step Size: {self.optim.param_groups[0]["step_size"]:.5f}')
        self.probmodel.reset_parameters()
        tune_chain = Chain(probmodel=self.probmodel)
        tune_chain.running_accepts.momentum = 0.5
        for tune_step in range(self.burn_in):
            sample_log_prob, sample = self.propose()
            accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
            tune_chain += (self.probmodel, sample_log_prob, accept)
            # if tune_step < self.burn_in and tune_step % tune_interval_length == 0 and tune_step > 0:
            if tune_step > 1:
                self.optim.dual_average_tune(tune_chain.accepts[-tune_interval_length:], tune_step,
                                             np.exp(log_ratio.item()))
            if not accept:
                if torch.isnan(sample_log_prob['log_prob']):
                    print(self.chain.state)
                    exit()
                self.probmodel.load_state_dict(self.chain.state['state_dict'])

    def sample_chain(self):
        self.probmodel.reset_parameters()
        if self.pretrain:
            try:
                self.probmodel.pretrain()
            except:
                print(
                    f'Tried pretraining but couldnt find a probmodel.pretrain() method ... Continuing wihtout pretraining.')
        if self.tune:
            self.tune_step_size()
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
    def __init__(self,
                 probmodel,
                 step_size=0.01,
                 num_steps=100,
                 num_chains=3,
                 burn_in=10,
                 pretrain=False,
                 tune=False):
        self.probmodel = probmodel
        self.chain = None
        self.num_chains = num_chains
        self.step_size = step_size
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.pretrain = pretrain
        self.tune = tune

    def sample_independent_chain(self):
        probmodel = copy.deepcopy(self.probmodel)
        probmodel.reset_parameters()
        if self.pretrain:
            probmodel.pretrain()
        optim = SGLD_Optim(probmodel, step_size=self.step_size, prior_std=0., addnoise=True)
        chain = Chain(probmodel=probmodel)
        for step in range(self.num_steps):
            sample_log_prob, sample = self.propose(probmodel, optim)
            accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], chain.state['log_prob'])
            chain += (probmodel, sample_log_prob, accept, step)
            if not accept:
                probmodel.load_state_dict(chain.state['state'])
        assert len(chain.accepted_steps) > self.burn_in, f'{len(chain.accepted_steps)=} <= {self.burn_in=}'
        chain.accepted_steps = chain.accepted_steps[self.burn_in:]
        return chain

    def sample_chain(self, step_size=None):
        if self.pretrain:
            self.probmodel.pretrain()
        self.optim = SGLD_Optim(self.probmodel,
                                step_size=step_size,
                                prior_std=0.,
                                addnoise=True)
        if self.tune: self.tune_step_size()
        self.chain = Chain(probmodel=self.probmodel)
        for step in range(self.num_steps):
            sample_log_prob, sample = self.propose()
            accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], self.chain.state['log_prob'])
            self.chain += (self.probmodel, sample_log_prob, accept, step)
            if not accept:
                self.probmodel.load_state_dict(self.chain.state['state'])
        print(len(self.chain))
        assert len(self.chain.accepted_steps) > self.burn_in, f'{len(self.chain.accepted_steps)=} <= {self.burn_in=}'
        self.chain.accepted_steps = self.chain.accepted_steps[self.burn_in:]

    def posterior_dist(self, param=None, verbose=False, plot=True):
        if len(self.probmodel.state_dict()) == 1:
            post = []
            accepted_models = self.chain.samples
            for model_state_dict in accepted_models:
                post.append(list(model_state_dict.values())[0])

        elif len(self.probmodel.state_dict()) > 1:
            param_names = list(self.probmodel.state_dict().keys())
            accepted_models = self.chain.samples
            for param_name in param_names:
                post = []
                for model_state_dict in accepted_models:
                    post.append(model_state_dict[param_name])
                post = torch.cat(post)

    def trace(self, param=None, verbose=False, plot=True):
        if len(self.probmodel.state_dict()) >= 1:
            param_names = list(self.probmodel.state_dict().keys())
            accepted_models = [self.chain.samples[idx] for idx in self.chain.accepted_steps]

            for param_name in param_names:
                post = []
                for model_state_dict in accepted_models:
                    post.append(model_state_dict[param_name])

    def sample_chains(self):
        if self.num_chains > 1:
            self.parallel_chains = [SGLD_Chain(copy.deepcopy(self.probmodel),
                                               step_size=self.step_size,
                                               num_steps=self.num_steps,
                                               burn_in=self.burn_in,
                                               pretrain=self.pretrain,
                                               tune=False)
                                    for _ in range(self.num_chains)]
            chains = Parallel(n_jobs=self.num_chains)(delayed(chain.sample_chain)() for chain in self.parallel_chains)
        elif self.num_chains == 1:
            chain = SGLD_Chain(copy.deepcopy(self.probmodel),
                               step_size=self.step_size,
                               num_steps=self.num_steps,
                               burn_in=self.burn_in,
                               pretrain=self.pretrain,
                               tune=False)
            chains = [chain.sample_chain()]
        self.chain = Chain(probmodel=self.probmodel)
        for chain in chains:
            self.chain += chain
        return chains

    def __str__(self):
        return 'SGLD'








if __name__ == "__main__":


    def generate_linear_regression_data(num_samples=100, m=1.0, b=-1.0, y_noise=1.0, x_noise=.01, plot=False):
        x = torch.linspace(-2, 2, num_samples).reshape(-1, 1)
        x += x_noise * torch.randn_like(x)
        y = m * x + b
        y += y_noise * torch.randn_like(y)
        return x, y


    class RegressionNNHomo(torch.nn.Module):
        def __init__(self, x, y, batch_size=1):
            super().__init__()
            self.data = x
            self.target = y
            self.dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=batch_size,
                                         drop_last=False)
            num_hidden = 50
            self.model = nn.Sequential(nn.Linear(1, num_hidden),
                                       nn.ReLU(),
                                       nn.Linear(num_hidden, num_hidden),
                                       nn.ReLU(),
                                       nn.Linear(num_hidden, 1))
            self.log_std = nn.Parameter(FloatTensor([-1]))

        def reset_parameters(self):
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            self.log_std.data = FloatTensor([3.])

        def sample(self):
            self.reset_parameters()

        def forward(self, x):
            pred = self.model(x)
            return pred

        def log_prob(self, data, target):
            mu = self.forward(data)
            mse = F.mse_loss(mu, target)
            log_prob = torch.distributions.Normal(mu, F.softplus(self.log_std)).log_prob(target).mean() * len(
                self.dataloader.dataset)
            return {'log_prob': log_prob, 'MSE': mse.detach_()}

    x, y = generate_linear_regression_data(num_samples=1000, m=-2., b=-1, y_noise=0.5)
    linreg = RegressionNNHomo(x, y, batch_size=50)
    sampler = SGLD_Sampler(probmodel=linreg, step_size=0.001, num_steps=500, burn_in=50)
    sampler.sample_chains()

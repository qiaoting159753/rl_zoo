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


class MCMC_Optim:
    def __init__(self):
        self.tune_params = {'delta': 0.65,
                            't0': 10,
                            'gamma': .05,
                            'kappa': .75,
                            # 'mu': np.log(self.param_groups[0]["step_size"]),
                            'mu': 0.,
                            'H': 0,
                            'log_eps': 1.}

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


class SGLD_Optim(Optimizer):
    """Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
    """

    def __init__(self,
                 params,
                 lr=0.001,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGLD_Optim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD_Optim, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])
                noise_std = torch.tensor([2 * group['lr']])
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0,
                                                          std=1) * noise_std
                p.data.add_(noise)

        return 1.0


# class SGLD_Optim(Optimizer, MCMC_Optim):
#     def __init__(self, model, step_size=0.1, prior_std=1., addnoise=True):
#         weight_decay = 1 / (prior_std ** 2) if prior_std != 0 else 0
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if step_size < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(step_size))
#         defaults = dict(step_size=step_size, weight_decay=weight_decay, addnoise=addnoise)
#         self.model = model
#         params = self.model.parameters()
#         Optimizer.__init__(self, params=params, defaults=defaults)
#         MCMC_Optim.__init__(self)
#
#     def step(self):
#         log_prob = None
#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 grad.clamp_(-1000, 1000)
#                 if weight_decay != 0:
#                     grad.add_(alpha=weight_decay, other=p.data)
#                 if group['addnoise']:
#                     noise = torch.randn_like(p.data).mul_(group['step_size'] ** 0.5)
#                     p.data.add_(grad, alpha=-0.5 * group['step_size'])
#                     p.data.add_(noise)
#                     if torch.isnan(p.data).any(): exit('Nan param')
#                     if torch.isinf(p.data).any(): exit('inf param')
#                 else:
#                     p.data.add_(other=0.5 * grad, alpha=-group['step_size'], )
#         return log_prob


class Sampler_Chain:
    def __init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune):
        self.probmodel = probmodel
        self.chain = Chain(probmodel=self.probmodel)
        self.step_size = step_size
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.pretrain = pretrain
        self.tune = tune

    def propose(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    # def tune_step_size(self):
    #     tune_interval_length = 100
    #     print(f'Tuning: Init Step Size: {self.optim.param_groups[0]["step_size"]:.5f}')
    #     self.probmodel.reset_parameters()
    #     tune_chain = Chain(probmodel=self.probmodel)
    #     tune_chain.running_accepts.momentum = 0.5
    #     for tune_step in range(self.burn_in):
    #         sample_log_prob, sample = self.propose()
    #         accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
    #         tune_chain += (self.probmodel, sample_log_prob, accept)
    #         # if tune_step < self.burn_in and tune_step % tune_interval_length == 0 and tune_step > 0:
    #         if tune_step > 1:
    #             # self.optim.dual_average_tune(tune_chain, np.exp(log_ratio.item()))
    #             self.optim.dual_average_tune(tune_chain.accepts[-tune_interval_length:], tune_step,
    #                                          np.exp(log_ratio.item()))
    #         # self.optim.tune(tune_chain.accepts[-tune_interval_length:])
    #         if not accept:
    #             if torch.isnan(sample_log_prob['log_prob']):
    #                 print(self.chain.state)
    #                 exit()
    #             self.probmodel.load_state_dict(self.chain.state['state_dict'])

    def sample_chain(self):
        self.probmodel.reset_parameters()
        self.chain = Chain(probmodel=self.probmodel)
        for step in range(self.num_steps):
            proposal_log_prob, sample = self.propose()
            # accept, log_ratio = self.acceptance(proposal_log_prob['log_prob'], self.chain.state['log_prob']['log_prob'])
            accept = True
            self.chain += (self.probmodel, proposal_log_prob, accept)
            if not accept:
                if torch.isnan(proposal_log_prob['log_prob']):
                    print("nnnnnnnnnannnannannana" + self.chain.state)
                    exit()
                self.probmodel.load_state_dict(self.chain.state['state_dict'])
        self.chain = self.chain[self.burn_in:]
        return self.chain


class SGLD_Chain(Sampler_Chain):
    def __init__(self, probmodel, step_size=0.0001, num_steps=2000, burn_in=100, pretrain=False, tune=False):
        Sampler_Chain.__init__(self, probmodel, step_size, num_steps, burn_in, pretrain, tune)
        self.optim = SGLD_Optim(self.probmodel.parameters())
        self.acceptance = SDE_Acceptance()

    def __repr__(self):
        return 'SGLD'

    @torch.enable_grad()
    def propose(self):
        self.optim.zero_grad()
        batch = next(self.probmodel.dataloader.__iter__())
        log_prob = self.probmodel.log_prob(*batch)
        (-log_prob['log_prob']).backward()
        self.optim.step()
        return log_prob, self.probmodel


















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


    # def sample_chain(self, step_size=None):
    #     if self.pretrain:
    #         self.probmodel.pretrain()
    #     self.optim = SGLD_Optim(self.probmodel,
    #                             step_size=step_size,
    #                             prior_std=0.,
    #                             addnoise=True)
    #     # if self.tune: self.tune_step_size()
    #     self.chain = Chain(probmodel=self.probmodel)
    #     for step in range(self.num_steps):
    #         sample_log_prob, sample = self.propose()
    #         accept, log_ratio = self.acceptance(sample_log_prob['log_prob'], self.chain.state['log_prob'])
    #         self.chain += (self.probmodel, sample_log_prob, accept, step)
    #         if not accept:
    #             self.probmodel.load_state_dict(self.chain.state['state'])
    #     assert len(self.chain.accepted_steps) > self.burn_in, f'{len(self.chain.accepted_steps)=} <= {self.burn_in=}'
    #     self.chain.accepted_steps = self.chain.accepted_steps[self.burn_in:]


class SGLD_Sampler(Sampler):
    def __init__(self, probmodel, step_size=0.01, num_steps=10000, num_chains=7, burn_in=500, pretrain=True, tune=True):
        Sampler.__init__(self, probmodel, step_size, num_steps, num_chains, burn_in, pretrain, tune)

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


        @torch.no_grad()
        def predict(self, chains, plot=True):
            x_min = 2 * self.data.min()
            x_max = 2 * self.data.max()
            data = torch.linspace(x_min, x_max, steps=100).reshape(-1, 1)
            def parallel_predict(parallel_chain):
                parallel_pred = []
                for model_state_dict in parallel_chain.samples[::50]:
                    self.load_state_dict(model_state_dict)
                    pred_mu_i = self.forward(data)
                    parallel_pred.append(pred_mu_i)
                try:
                    parallel_pred_mu = torch.stack(
                        parallel_pred)  # list [ pred_0, pred_1, ... pred_N] -> Tensor([pred_0, pred_1, ... pred_N])
                    return parallel_pred_mu
                except:
                    pass
            parallel_pred = Parallel(n_jobs=len(chains))(delayed(parallel_predict)(chain) for chain in chains)
            pred = [parallel_pred_i for parallel_pred_i in parallel_pred if
                    parallel_pred_i is not None]  # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]
            # pred_log_std = [parallel_pred_i for parallel_pred_i in parallel_pred_log_std if parallel_pred_i is not None] # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]
            pred = torch.cat(pred).squeeze()  # cat list of tensors to single prediciton tensor with samples in first dim
            std = F.softplus(self.log_std)
            epistemic = pred.std(dim=0)
            aleatoric = std
            total_std = (epistemic ** 2 + aleatoric ** 2) ** 0.5
            mu = pred.mean(dim=0)
            std = std.mean(dim=0)
            data.squeeze_()
            if plot:
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
                axs = axs.flatten()

                axs[0].scatter(self.data, self.target, alpha=1, s=1, color='blue')
                axs[0].plot(data.squeeze(), mu, alpha=1., color='red')
                axs[0].fill_between(data, mu + total_std, mu - total_std, color='red', alpha=0.25)
                axs[0].fill_between(data, mu + 2 * total_std, mu - 2 * total_std, color='red', alpha=0.10)
                axs[0].fill_between(data, mu + 3 * total_std, mu - 3 * total_std, color='red', alpha=0.05)

                # [axs[1].plot(data, pred, alpha=0.1, color='red') for pred in pred]
                # axs[1].scatter(self.data, self.target, alpha=1, s=1, color='blue')
                #
                # axs[2].scatter(self.data, self.target, alpha=1, s=1, color='blue')
                # axs[2].plot(data, mu, color='red')
                # axs[2].fill_between(data, mu - aleatoric, mu + aleatoric, color='red', alpha=0.25, label='Aleatoric')
                # axs[2].legend()

                axs[3].scatter(self.data, self.target, alpha=1, s=1, color='blue')
                axs[3].plot(data, mu, color='red')
                axs[3].fill_between(data, mu - epistemic, mu + epistemic, color='red', alpha=0.25, label='Epistemic')
                axs[3].legend()

                plt.ylim(2 * self.target.min(), 2 * self.target.max())
                plt.xlim(x_min, x_max)
                plt.show()

    x, y = generate_linear_regression_data(num_samples=1000, m=-2., b=-1, y_noise=0.5)
    linreg = RegressionNNHomo(x, y, batch_size=50)
    sampler = SGLD_Sampler(probmodel=linreg, step_size=0.001, num_steps=100, burn_in=30)
    chains = sampler.sample_chains()
    linreg.predict(chains)
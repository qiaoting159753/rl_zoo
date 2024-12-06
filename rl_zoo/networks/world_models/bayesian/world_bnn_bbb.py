import torch
import numpy as np
from torch import optim
from rl_zoo.networks.world_models import World_Model
from rl_zoo.utils import normalize_observation, denormalize_observation_delta, normalize_observation_delta
import torchbnn as bnn
import torch.nn as nn
import torch.nn.functional as F
from .bnn_bbb_torchbnn import BayesLinear
import math


def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1):
    """
    An method for calculating KL divergence between two Normal distribtuion.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).

    """
    kl = log_sigma_1 - log_sigma_0 + \
         (torch.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2) / (2 * math.exp(log_sigma_1) ** 2) - 0.5
    return kl.sum()

def bayesian_kl_loss(model, reduction='mean', last_layer_only=False):
    """
    An method for calculating KL divergence of whole layers in the model.


    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.

    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for m in model.modules():
        if isinstance(m, (BayesLinear)):
            kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
            kl_sum += kl
            n += len(m.weight_mu.view(-1))

            if m.bias:
                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))

    if last_layer_only or n == 0:
        return kl

    if reduction == 'mean':
        return kl_sum / n
    elif reduction == 'sum':
        return kl_sum
    else:
        raise ValueError(reduction + " is not valid")



class BNN_World(nn.Module):
    def __init__(self, observation_size, num_actions, hidden_size, sigma):
        super().__init__()
        self.nn_layers = nn.ModuleList()

        self.l1 = BayesLinear(prior_mu=0, prior_sigma=sigma, in_features=(observation_size + num_actions),out_features=hidden_size[0])
        self.l2 = BayesLinear(prior_mu=0, prior_sigma=sigma, in_features=hidden_size[0], out_features=hidden_size[1])
        self.l3 = BayesLinear(prior_mu=0, prior_sigma=sigma, in_features=hidden_size[1], out_features=2 * observation_size)
        self.add_module('l1',self.l1)
        self.add_module('l2', self.l2)
        self.add_module('l3', self.l3)
        self.nn_layers.append(self.l1)
        self.nn_layers.append(self.l2)
        self.nn_layers.append(self.l3)


        # self.register_parameter('l1', self.l1.parameters())
        # self.register_parameter('l2', self.l2.parameters())
        # self.register_parameter('l3', self.l3.parameters())
    def forward(self, in_data, sample):
        x = self.l1(in_data, sample)
        x = F.relu(x)
        x = self.l2(x, sample)
        x = F.relu(x)
        x = self.l3(x, sample)
        return x

class Bayesian_World_Model_BBB(World_Model):
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 device: str,
                 ratio: float,
                 sigma: float = 0.1,
                 l_r: float = 0.001,
                 hidden_size=None,
                 sas: bool = True,
                 prob_rwd: bool = True):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        if hidden_size is None:
            hidden_size = [128, 128]
        self.statistics = None
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.l_r = l_r
        self.ratio = ratio
        self.device = device
        self.sas = sas
        self.prob_rwd = prob_rwd
        self.world_model =BNN_World(observation_size, num_actions, [128, 128], sigma=sigma)
        self.kl_loss = bnn.BKLLoss()
        self.world_optimizers = optim.Adam(self.world_model.parameters(), lr=self.l_r)
        self.world_model.to(self.device)

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        target = next_states - states
        normalized_target = normalize_observation_delta(target, self.statistics)
        normlized_state = normalize_observation(states, self.statistics)
        x = torch.cat((normlized_state, actions), dim=1)
        preds = self.world_model(x, sample=True)
        mean_pred = preds[:, :self.observation_size]
        var_pred = preds[:, self.observation_size:]
        var_pred = torch.tanh(var_pred)
        var_pred = torch.exp(var_pred)
        self.world_optimizers.zero_grad()
        kl_loss = bayesian_kl_loss(self.world_model)
        nll_loss = F.gaussian_nll_loss(input=mean_pred, target=normalized_target, var=var_pred).mean()
        loss = nll_loss + self.ratio * kl_loss
        loss.backward()
        self.world_optimizers.step()

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor, train_reward:bool
    ) -> tuple[float, float, torch.Tensor]:
        normlized_state = normalize_observation(observation, self.statistics)
        x = torch.cat((normlized_state, actions), dim=1)
        mean_s = []
        var_s = []
        sample_world_times = 20
        # for _ in range(sample_times):
        for i in range(sample_world_times):
            pred = self.world_model(x, sample=True)
            mean_pred = pred[:, :self.observation_size]
            var_pred = pred[:, self.observation_size:]
            var_pred = torch.tanh(var_pred)
            var_pred = torch.exp(var_pred)
            mean_s.append(mean_pred)
            var_s.append(var_pred)
        all_vars = torch.vstack(var_s).squeeze()
        aleatoric = all_vars.detach().cpu().numpy()
        all_means = torch.vstack(mean_s).squeeze()
        epistemic = all_means.detach().cpu().numpy()
        aleatoric = (aleatoric ** 2).mean(axis=0) ** 0.5
        epistemic = epistemic.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc).item()

        uncert_rwd = 0.0
        samples = None
        if not train_reward:
            sample_times = 20
            dist = torch.distributions.Normal(all_means, all_vars)
            samples = dist.sample([sample_times])
            samples = torch.reshape(samples, (sample_times * sample_world_times, self.observation_size))
            samples = denormalize_observation_delta(samples, self.statistics)
            observationss = torch.repeat_interleave(observation, repeats=sample_times * sample_world_times, dim=0)
            samples += observationss
        else:
            # Reward Uncertainty
            sample_times = 20
            mean_s = torch.vstack(mean_s)
            vars_s = torch.vstack(var_s)
            dist = torch.distributions.Normal(mean_s, vars_s)
            samples = dist.sample([sample_times])
            samples = torch.reshape(samples, (sample_times * sample_world_times, self.observation_size))
            samples = denormalize_observation_delta(samples, self.statistics)
            observationss = torch.repeat_interleave(observation, repeats=sample_times * sample_world_times, dim=0)
            actionss = torch.repeat_interleave(actions, repeats=sample_times * sample_world_times, dim=0)
            samples += observationss

            if self.sas:
                if self.prob_rwd:
                    rewards, rwd_var = self.reward_network(observationss, actionss, samples)
                    epis_uncert = torch.var(rewards, dim=0).item()
                    rwd_var = rwd_var.squeeze().detach().cpu().numpy()
                    alea_uncert = (rwd_var ** 2).mean(axis=0) ** 0.5
                    epis_uncert = np.minimum(epis_uncert, 10e3)
                    alea_uncert = np.minimum(alea_uncert, 10e3)
                    uncert_rwd = ((epis_uncert ** 2) + (alea_uncert ** 2)) ** 0.5
                else:
                    rewards = self.reward_network(observationss, actionss, samples)
                    uncert_rwd = torch.var(rewards, dim=0).item()
            else:
                if self.prob_rwd:
                    rewards, rwd_var = self.reward_network(samples, actionss)
                    epis_uncert = torch.var(rewards, dim=0).item()
                    rwd_var = rwd_var.squeeze().detach().cpu().numpy()
                    alea_uncert = (rwd_var ** 2).mean(axis=0) ** 0.5
                    epis_uncert = np.minimum(epis_uncert, 10e3)
                    alea_uncert = np.minimum(alea_uncert, 10e3)
                    uncert_rwd = ((epis_uncert ** 2) + (alea_uncert ** 2)) ** 0.5
                else:
                    rewards = self.reward_network(samples, actionss)
                    uncert_rwd = torch.var(rewards, dim=0).item()

        return uncert, uncert_rwd, samples

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normlized_state = normalize_observation(observation, self.statistics)
        normalized_obs_a = torch.cat((normlized_state, actions), dim=1)
        pred = self.world_model(normalized_obs_a, sample=False)
        preds = pred[:, :self.observation_size]
        mean_deltas = denormalize_observation_delta(preds, self.statistics)
        preds = mean_deltas + observation
        return preds, None, None, None

    def train_together(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, ):
        normlized_state = normalize_observation(states, self.statistics)
        x = torch.cat((normlized_state, actions), dim=1)
        mean_s = []
        var_s = []
        act_s = []
        state_s = []
        rwd_s = []
        sample_world_times = 5
        # for _ in range(sample_times):
        for i in range(sample_world_times):
            pred = self.world_model(x)
            mean_pred = pred[:, :self.observation_size]
            var_pred = pred[:, self.observation_size:]
            var_pred = torch.tanh(var_pred)
            var_pred = torch.exp(var_pred)
            mean_s.append(mean_pred)
            var_s.append(var_pred)
            act_s.append(actions)
            state_s.append(states)
            rwd_s.append(rewards)
        mean_s = torch.vstack(mean_s)
        vars_s = torch.vstack(var_s)
        act_s = torch.vstack(act_s)
        state_s = torch.vstack(state_s)
        rwd_s = torch.vstack(rwd_s)

        sample_times = 20
        dist = torch.distributions.Normal(mean_s, vars_s)
        samples = dist.sample([sample_times])

        state_s = torch.repeat_interleave(state_s.unsqueeze(dim=0), sample_times, dim=0)
        act_s = torch.repeat_interleave(act_s.unsqueeze(dim=0), sample_times, dim=0)
        rwd_s = torch.repeat_interleave(rwd_s.unsqueeze(dim=0), sample_times, dim=0)

        act_s = torch.reshape(act_s, (act_s.shape[0] * act_s.shape[1], act_s.shape[2]))
        state_s = torch.reshape(state_s, (state_s.shape[0] * state_s.shape[1], state_s.shape[2]))
        rwd_s = torch.reshape(rwd_s, (rwd_s.shape[0] * rwd_s.shape[1], rwd_s.shape[2]))
        samples = torch.reshape(samples, (samples.shape[0] * samples.shape[1], samples.shape[2]))

        samples = denormalize_observation_delta(samples, self.statistics)
        samples += state_s
        samples = samples.detach()

        if self.prob_rwd:
            if self.sas:
                rwd_mean, rwd_var = self.reward_network(state_s, act_s, samples)
            else:
                rwd_mean, rwd_var = self.reward_network(samples, act_s)
            rwd_loss = F.gaussian_nll_loss(rwd_mean, rwd_s, rwd_var)
        else:
            if self.sas:
                rwd_mean = self.reward_network(state_s, act_s, samples)
            else:
                rwd_mean = self.reward_network(samples, act_s)
            rwd_loss = F.mse_loss(rwd_mean, rwd_s)
        self.reward_optimizer.zero_grad()
        rwd_loss.backward()
        self.reward_optimizer.step()

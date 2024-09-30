from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, denormalize_observation_delta
import logging
import sys
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)


def softmax_CE_preact_hessian(last_layer_acts):
    side = last_layer_acts.shape[1]
    I = torch.eye(side).type(torch.ByteTensor)
    # for i != j    H = -ai * aj -- Note that these are activations not pre-activations
    Hl = - last_layer_acts.unsqueeze(1) * last_layer_acts.unsqueeze(2)
    # for i == j    H = ai * (1 - ai)
    Hl[:, I] = last_layer_acts * (1 - last_layer_acts)
    return Hl


def layer_act_hessian_recurse(prev_hessian, prev_weights, layer_pre_acts):
    newside = layer_pre_acts.shape[1]
    batch_size = layer_pre_acts.shape[0]
    I = torch.eye(newside).type(torch.ByteTensor)  # .unsqueeze(0).expand([batch_size, -1, -1])
    #     print(d_act(layer_pre_acts).unsqueeze(1).shape, I.shape)
    B = prev_weights.data.new(batch_size, newside, newside).fill_(0)
    B[:, I] = (layer_pre_acts > 0).type(B.type())  # d_act(layer_pre_acts)
    D = prev_weights.data.new(batch_size, newside, newside).fill_(0)  # is just 0 for a piecewise linear
    #     D[:, I] = dd_act(layer_pre_acts) * act_grads
    Hl = torch.bmm(torch.t(prev_weights).unsqueeze(0).expand([batch_size, -1, -1]), prev_hessian)
    Hl = torch.bmm(Hl, prev_weights.unsqueeze(0).expand([batch_size, -1, -1]))
    Hl = torch.bmm(B, Hl)
    Hl = torch.matmul(Hl, B)
    Hl = Hl + D
    return Hl


def chol_scale_invert_kron_factor(factor, prior_scale, data_scale, upper=False):
    scaled_factor = data_scale * factor + prior_scale * torch.eye(factor.shape[0]).type(factor.type())
    inv_factor = torch.inverse(scaled_factor)
    chol_inv_factor = torch.linalg.cholesky(inv_factor).mH
    return chol_inv_factor


def sample_K_laplace_MN(MAP, upper_Qinv, lower_HHinv):
    # H = Qi (kron) HHi
    # sample isotropic unit variance mtrix normal
    Z = MAP.data.new(MAP.size()).normal_(mean=0, std=1)
    # AAT = HHi
    #     A = torch.cholesky(HHinv, upper=False)
    # BTB = Qi
    #     B = torch.cholesky(Qinv, upper=True)
    all_mtx_sample = MAP + torch.matmul(torch.matmul(lower_HHinv, Z), upper_Qinv)
    weight_mtx_sample = all_mtx_sample[:, :-1]
    bias_mtx_sample = all_mtx_sample[:, -1]
    return weight_mtx_sample, bias_mtx_sample


class Linear_2L_KFRA(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_2L_KFRA, self).__init__()
        self.n_hid = n_hid
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(self.n_hid, output_dim)
        # choose your non linearity
        self.act = nn.ReLU(inplace=True)
        self.one = None
        self.a2 = None
        self.h2 = None
        self.a1 = None
        self.h1 = None
        self.a0 = None

    def forward(self, x):
        self.one = x.new(x.shape[0], 1).fill_(1)
        a0 = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        self.a0 = torch.cat((a0.data, self.one), dim=1)
        # -----------------
        h1 = self.fc1(a0)
        self.h1 = h1.data  # torch.cat((h1, self.one), dim=1)
        # -----------------
        a1 = self.act(h1)
        #         a1.retain_grad()
        self.a1 = torch.cat((a1.data, self.one), dim=1)
        # -----------------
        h2 = self.fc2(a1)
        self.h2 = h2.data  # torch.cat((h2, self.one), dim=1)
        # -----------------
        a2 = self.act(h2)
        #         a2.retain_grad()
        self.a2 = torch.cat((a2.data, self.one), dim=1)
        # -----------------
        h3 = self.fc3(a2)
        return h3

    def sample_predict(self, x, Nsamples, Qinv1, HHinv1, MAP1, Qinv2, HHinv2, MAP2, Qinv3, HHinv3, MAP3):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        x = x.view(-1, self.input_dim)
        for i in range(Nsamples):
            # -----------------
            w1, b1 = sample_K_laplace_MN(MAP1, Qinv1, HHinv1)
            a = torch.matmul(x, torch.t(w1)) + b1.unsqueeze(0)
            a = self.act(a)
            # -----------------
            w2, b2 = sample_K_laplace_MN(MAP2, Qinv2, HHinv2)
            a = torch.matmul(a, torch.t(w2)) + b2.unsqueeze(0)
            a = self.act(a)
            # -----------------
            w3, b3 = sample_K_laplace_MN(MAP3, Qinv3, HHinv3)
            y = torch.matmul(a, torch.t(w3)) + b3.unsqueeze(0)
            predictions[i] = y
        return predictions


class Bayesian_World_Model_Laplace_JA(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r,
                 hidden_size,
                 device):
        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model = Linear_2L_KFRA(input_dim=observation_size + num_actions, output_dim=2 * observation_size,
                                          n_hid=128)
        self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r, betas=(0.9, 0.999), eps=1e-08)
        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)
        self.reward_optimizers = torch.optim.Adam(self.reward_model.parameters(), lr=l_r)
        self.reward_model.to(self.device)
        self.world_model.to(self.device)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5,
        #                                  weight_decay=(1 / self.prior_sig ** 2))

    def set_statistics(self, statistics: dict) -> None:
        """
        Update all statistics for normalization for all world models and the
        ensemble itself.

        :param (Dictionary) statistics:
        """
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                statistics[key] = torch.FloatTensor(statistics[key]).to(self.device)
        self.statistics = statistics
        self.world_model.statistics = statistics

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        s_n_a = torch.cat((states, actions), dim=1)
        pred_s = self.world_model.forward(s_n_a)
        n_mean_delta = pred_s[:, self.observation_size:]
        n_log_delta = pred_s[:, :self.observation_size]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        model_loss = F.gaussian_nll_loss(input=n_mean_delta, target=delta_targets_normalized, var=normalized_var).mean()
        self.world_optimizers.zero_grad()
        model_loss.backward()
        self.world_optimizers.step()

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_n_a = torch.cat((observation, actions), dim=1)
        pred_s = self.world_model.forward(s_n_a)
        n_mean_delta = pred_s[:, self.observation_size:]
        n_log_delta = pred_s[:, :self.observation_size]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        if torch.any(torch.isnan(n_mean_delta)):
            logging.info("Predicting all Nans")
            sys.exit()
        prediction = denormalize_observation_delta(n_mean_delta, self.statistics)
        prediction += observation
        return prediction, None, n_mean_delta, normalized_var

    def pred_rewards(self, observation: torch.Tensor,
                     action: torch.Tensor, next_observation: torch.Tensor):
        """
        predict reward based on current observation and action and next state
        """
        pred_reward, reward_var = self.reward_model.forward(observation, action, next_observation)
        return pred_reward, None, reward_var

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        """
        Train the reward with S, A, SN to eliminate difference between them.
        Args:
            states:
            actions:
            next_states:
            rewards:
        """
        self.reward_optimizers.zero_grad()
        rwd_mean, rwd_var = self.reward_model.forward(states, actions, next_states)
        # reward_loss = F.mse_loss(rwd_mean, sub_rewards)
        reward_loss = F.gaussian_nll_loss(input=rwd_mean, target=rewards, var=rwd_var).mean()
        reward_loss.backward()
        self.reward_optimizers.step()

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:

        self.world_model.eval()
        it_counter = 0
        cum_HH1 = self.world_model.fc1.weight.data.new(self.world_model.n_hid, self.world_model.n_hid).fill_(0)
        cum_HH2 = self.world_model.fc1.weight.data.new(self.world_model.n_hid, self.world_model.n_hid).fill_(0)
        cum_HH3 = self.world_model.fc1.weight.data.new(self.world_model.output_dim, self.world_model.output_dim).fill_(0)
        cum_Q1 = self.world_model.fc1.weight.data.new(self.world_model.input_dim + 1, self.world_model.input_dim + 1).fill_(0)
        cum_Q2 = self.world_model.fc1.weight.data.new(self.world_model.n_hid + 1, self.world_model.n_hid + 1).fill_(0)
        cum_Q3 = self.world_model.fc1.weight.data.new(self.world_model.n_hid + 1, self.world_model.n_hid + 1).fill_(0)
        # Forward pass
        x = torch.cat((observation, actions), dim=1)
        self.world_optimizers.zero_grad()
        pred_s = self.world_model(x)
        out_act = F.softmax(pred_s, dim=1)

        n_mean_delta = pred_s[:, self.observation_size:]
        n_log_delta = pred_s[:, :self.observation_size]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)

        loss = F.gaussian_nll_loss(input=n_mean_delta, target=n_mean_delta, var=normalized_var).mean()
        loss.backward()

        #     ------------------------------------------------------------------
        HH3 = softmax_CE_preact_hessian(out_act.data)
        cum_HH3 += HH3.sum(dim=0)
        #     print(model.a2.data.shape)
        Q3 = torch.bmm(self.world_model.a2.data.unsqueeze(2), self.world_model.a2.data.unsqueeze(1))
        cum_Q3 += Q3.sum(dim=0)
        #     ------------------------------------------------------------------
        HH2 = layer_act_hessian_recurse(prev_hessian=HH3, prev_weights=self.world_model.fc3.weight.data,
                                        layer_pre_acts=self.world_model.h2.data)
        cum_HH2 += HH2.sum(dim=0)
        Q2 = torch.bmm(self.world_model.a1.data.unsqueeze(2), self.world_model.a1.data.unsqueeze(1))
        cum_Q2 += Q2.sum(dim=0)
        #     ------------------------------------------------------------------
        HH1 = layer_act_hessian_recurse(prev_hessian=HH2, prev_weights=self.world_model.fc2.weight.data,
                                        layer_pre_acts=self.world_model.h1.data)
        cum_HH1 += HH1.sum(dim=0)
        Q1 = torch.bmm(self.world_model.a0.data.unsqueeze(2), self.world_model.a0.data.unsqueeze(1))
        cum_Q1 += Q1.sum(dim=0)
        #     ------------------------------------------------------------------
        it_counter += x.shape[0]
        # print(it_counter)

        EHH3 = cum_HH3 / it_counter
        EHH2 = cum_HH2 / it_counter
        EHH1 = cum_HH1 / it_counter
        EQ3 = cum_Q3 / it_counter
        EQ2 = cum_Q2 / it_counter
        EQ1 = cum_Q1 / it_counter
        MAP3 = torch.cat((self.world_model.fc3.weight.data, self.world_model.fc3.bias.data.unsqueeze(1)), dim=1)
        MAP2 = torch.cat((self.world_model.fc2.weight.data, self.world_model.fc2.bias.data.unsqueeze(1)), dim=1)
        MAP1 = torch.cat((self.world_model.fc1.weight.data, self.world_model.fc1.bias.data.unsqueeze(1)), dim=1)

        prior_prec = 0.001
        prior_scale = np.sqrt(prior_prec)
        data_scale = 0.001

        scale_inv_EQ1 = chol_scale_invert_kron_factor(EQ1, prior_scale, data_scale, upper=True)
        scale_inv_EHH1 = chol_scale_invert_kron_factor(EHH1, prior_scale, data_scale, upper=False)

        scale_inv_EQ2 = chol_scale_invert_kron_factor(EQ2, prior_scale, data_scale, upper=True)
        scale_inv_EHH2 = chol_scale_invert_kron_factor(EHH2, prior_scale, data_scale, upper=False)

        scale_inv_EQ3 = chol_scale_invert_kron_factor(EQ3, prior_scale, data_scale, upper=True)
        scale_inv_EHH3 = chol_scale_invert_kron_factor(EHH3, prior_scale, data_scale, upper=False)

        out = self.world_model.sample_predict(x, 10, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2,
                                        MAP2, scale_inv_EQ3, scale_inv_EHH3, MAP3)
        pred_s = torch.squeeze(out)

        n_mean_delta = pred_s[:, self.observation_size:]
        n_log_delta = pred_s[:, :self.observation_size]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        sample1 = torch.distributions.Normal(n_mean_delta, normalized_var).sample([10])
        sample1 = torch.reshape(sample1, (100, self.observation_size))

        prediction = denormalize_observation_delta(sample1, self.statistics)
        prediction += observation

        uncert = torch.sum(torch.var(prediction, dim=0))
        return uncert.item(), 0.0








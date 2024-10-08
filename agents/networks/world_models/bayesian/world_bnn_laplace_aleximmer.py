from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, denormalize_observation_delta
from .bayesian_aleximmer_la import KronLaplace
import torch.nn as nn


class CustomizedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = nn.Linear(input_size, next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size
        self.obs_length = int(output_size / 2)
        self.output_layer = nn.Linear(input_size, output_size)

    def customized_forward(self, state):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        return output

    def forward(self, state):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        mean_out = output[:, :self.obs_length]
        var_out = output[:, self.obs_length:]
        logvar = torch.tanh(var_out)
        normalized_var = torch.exp(logvar)
        output = torch.concatenate((mean_out, normalized_var), dim=1)
        return output


class Bayesian_World_Model_Laplace_AX(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r,
                 hidden_size,
                 sigma,
                 temperature,
                 prior_precision,
                 device,
                 sas: bool = True,
                 prob_rwd:bool = False):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        self.sas = sas
        self.prob_rwd = prob_rwd
        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model = CustomizedMLP(input_size=(observation_size + num_actions), output_size=2 * observation_size,
                                         hidden_sizes=[256, 256, 256])
        self.bnn = KronLaplace(model=self.world_model, likelihood="regression", sigma_noise=sigma,
                               temperature=temperature, prior_precision=prior_precision)
        self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r)
        self.world_model.to(self.device)

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train all

        :param states:
        :param actions:
        :param next_states:
        """
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        s_n_a = torch.cat((states, actions), dim=1)
        pred_s = self.world_model.customized_forward(s_n_a)
        n_mean_delta = pred_s[:, :self.observation_size]
        n_log_delta = pred_s[:, self.observation_size:]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        model_loss = F.gaussian_nll_loss(input=n_mean_delta, target=delta_targets_normalized, var=normalized_var).mean()
        self.world_optimizers.zero_grad()
        model_loss.backward()
        self.world_optimizers.step()

        pred_s = self.world_model.customized_forward(s_n_a)
        x = pred_s[:, :self.observation_size]
        y_x = ((delta_targets_normalized - x) ** 2).detach()
        self.bnn.fit((s_n_a, torch.cat((delta_targets_normalized, y_x), dim=1)))
        self.bnn.optimize_prior_precision(pred_type="glm")

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        uncert = 0.0
        if self.bnn.n_data > 0:
            s_n_a = torch.cat((observation, actions), dim=1)
            pred = self.bnn(s_n_a, pred_type="glm", link_approx="probit")
            mean, var = pred
            mean = mean.detach().cpu().numpy()
            var_all = torch.diagonal(var.squeeze()).unsqueeze(dim=0).detach().cpu().numpy()
            # Aleatoric: Last part of the mean
            # Epistemic: First/Second half of the var.
            epistemic = var_all[:, :self.observation_size]
            noises = mean[:, self.observation_size:]
            aleatoric = (noises ** 2).mean(axis=0) ** 0.5
            # epistemic = all_means.var(axis=0) ** 0.5
            aleatoric = np.minimum(aleatoric, 10e3)
            epistemic = np.minimum(epistemic, 10e3)
            total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
            uncert = np.mean(total_unc).item()

            # Sampling does not working well.
            # var_all = var_all.unsqueeze(dim=0)
            # dist1 = torch.distributions.Normal(mean, var_all)
            # # [100, 34]
            # first_sample = dist1.sample([20])
            # first_sample = first_sample.squeeze()
            # var_mean = first_sample[:, :self.observation_size]
            # var_var = first_sample[:, self.observation_size:]
            # var_var[var_var<0.00001] = 0.00001
            # dist2 = torch.distributions.Normal(var_mean, var_var)
            # second_sample = dist2.sample([20])
            # second_sample = second_sample.squeeze()
            # second_sample = torch.reshape(second_sample, (400, self.observation_size))
            # uncert = torch.mean(torch.var(second_sample, dim=0)).item()
        return uncert, 0.0

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_n_a = torch.cat((observation, actions), dim=1)
        pred_s = self.world_model.customized_forward(s_n_a)
        n_mean_delta = pred_s[:, :self.observation_size]
        prediction = denormalize_observation_delta(n_mean_delta, self.statistics)
        prediction += observation
        return prediction, None, None, None

from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from laplace import Laplace
from torch.utils.data import DataLoader, TensorDataset
from rl_zoo.networks.world_models import World_Model
from rl_zoo.utils.helpers import normalize_observation_delta, normalize_observation, denormalize_observation_delta

class PNN_MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        self.half_output = int(output_size / 2)
        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = nn.Linear(input_size, next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, state):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        left_output = output[:, :self.half_output]
        right_output = output[:, self.half_output:]
        right_output = torch.tanh(right_output)
        right_output = torch.exp(right_output)
        output = torch.cat((left_output, right_output), dim=1)
        return output


class Bayesian_World_Model_LA(World_Model):
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 device: str,
                 l_r: float = 0.001,
                 hidden_size=None,
                 temperature: float = 1.0,
                 prior_precision: float = 1.0,
                 sas: bool = True,
                 prob_rwd: bool = True):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        if hidden_size is None:
            hidden_size = [128, 128]
        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model = PNN_MLP(observation_size + num_actions, hidden_size, 2 * observation_size)
        self.world_model.to(device)
        self.dyna_optimizer = torch.optim.Adam(self.world_model.parameters(), l_r)
        self.l_a = Laplace(self.world_model,
                           "regression",
                           subset_of_weights="all",
                           hessian_structure="kron",
                           temperature=0.1,
                           prior_precision=prior_precision)

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_observation = normalize_observation(observation, self.statistics)
        s_n_a = torch.cat((normalized_observation, actions), dim=1)
        pred_s = self.world_model(s_n_a)
        n_mean = pred_s[:, :self.observation_size]
        prediction = denormalize_observation_delta(n_mean, self.statistics)
        prediction += observation
        return prediction, None, None, None

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        normalized_state = normalize_observation(states, self.statistics)
        s_n_a = torch.cat((normalized_state, actions), dim=1)
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        pred_s = self.world_model(s_n_a)
        mean_s = pred_s[:, :self.observation_size]
        var_s = pred_s[:, self.observation_size:]
        model_loss = F.gaussian_nll_loss(input=mean_s, target=delta_targets_normalized, var=var_s).mean()
        self.dyna_optimizer.zero_grad()
        model_loss.backward()
        self.dyna_optimizer.step()

        pred_s_2 = self.world_model(s_n_a)
        mean_s_2 = pred_s_2[:, :self.observation_size]
        y_x = (delta_targets_normalized - mean_s_2) ** 2
        target = torch.cat((delta_targets_normalized, y_x), dim=1).detach()

        train_loader = DataLoader(TensorDataset(s_n_a, target), batch_size=s_n_a.shape[0])
        self.l_a.fit(train_loader, override=False)

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor, train_reward:bool
    ) -> tuple[float, float]:
        uncert = 0.0
        rwd_uncert = 0.0
        if self.l_a.n_data > 0:
            normalized_state = normalize_observation(observation, self.statistics)
            s_n_a = torch.cat((normalized_state, actions), dim=1)
            f_mean, f_var = self.l_a(s_n_a, pred_type="glm", link_approx="mc", n_samples=100)
            f_mean_detach = f_mean.detach().squeeze().cpu().numpy()
            aleatoric = f_mean_detach[self.observation_size:]
            aleatoric = (aleatoric ** 2) ** 0.5
            f_var = f_var.squeeze()
            f_var = torch.diagonal(f_var)
            # Definitely not self.l_a.sigma_noise.item(), coz it remain the same everytime.
            epistemic = f_var[:self.observation_size].squeeze().detach().cpu().numpy()
            epistemic = epistemic ** 0.5
            # # epistemic = all_means.var(axis=0) ** 0.5
            aleatoric = np.minimum(aleatoric, 10e3)
            epistemic = np.minimum(epistemic, 10e3)
            total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
            uncert = np.mean(total_unc).item()

            if not train_reward:
                dist1 = torch.distributions.Normal(a, f_var)

        return uncert, rwd_uncert, None

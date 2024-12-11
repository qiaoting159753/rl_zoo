from __future__ import division
import copy
import torch
import torch.nn.functional as F
from rl_zoo.networks.world_models import World_Model
from rl_zoo.utils.helpers import denormalize_observation_delta, normalize_observation, normalize_observation_delta
from rl_zoo.networks.world_models.bayesian.bayesian_sgld_classes import SGLD_Sampler
from rl_zoo.networks.world_models.bayesian.bayesian_sgld_stationary import CustomizedMLP
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Bayesian_World_Model_SGLD(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 device,
                 l_r: float = 0.001,
                 hidden_size=None,
                 sas: bool = True,
                 prob_rwd: bool = True):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        if hidden_size is None:
            hidden_size = [128, 128]
        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model_0 = CustomizedMLP(observation_size=observation_size,
                                           num_actions=num_actions,
                                           device=device,
                                           hidden_sizes=hidden_size,
                                           name="model_0")
        self.world_model_1 = CustomizedMLP(observation_size=observation_size,
                                           num_actions=num_actions,
                                           device=device,
                                           hidden_sizes=hidden_size,
                                           name="model_1")
        self.world_model_0.to(self.device)
        self.world_model_1.to(self.device)

        self.world_optimizers = torch.optim.Adam(self.world_model_0.parameters(), lr=l_r)
        self.trained_once = False
        self.counter = 0
        self.stack_layers = 5
        self.data = torch.ones((self.stack_layers * 256, observation_size + num_actions))
        self.target = torch.ones((self.stack_layers * 256, observation_size))

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Predict delta
        normalized_observation = normalize_observation(observation, self.statistics)
        data = torch.cat((normalized_observation, actions), dim=1)
        n_mean, n_var = self.world_model_0.forward(data)
        prediction = denormalize_observation_delta(n_mean, self.statistics)
        prediction += observation
        return prediction, None, n_mean, n_var

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_state = normalize_observation(states, self.statistics)
        data = torch.cat((normalized_state, actions), dim=1)
        n_mean, n_var = self.world_model_0.forward(data)
        self.world_optimizers.zero_grad()
        model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
        model_loss.backward()
        self.world_optimizers.step()

        self.data[256 * (self.counter % self.stack_layers):256 * ((self.counter % self.stack_layers) + 1),
        :] = copy.deepcopy(data)
        self.target[256 * (self.counter % self.stack_layers):256 * ((self.counter % self.stack_layers) + 1),
        :] = copy.deepcopy(delta_targets_normalized)
        if self.counter > self.stack_layers:
            self.world_model_1.data = copy.deepcopy(self.data)
            self.world_model_1.data.to(self.device)
            self.world_model_1.target = copy.deepcopy(self.target)
            self.world_model_1.target.to(self.device)
            self.trained_once = True
            self.world_model_1.dataloader = DataLoader(TensorDataset(self.data, self.target), batch_size=50)
        self.counter += 1

    def estimate_uncertainty(self, observation: torch.Tensor, actions: torch.Tensor, train_reward:bool
    ) -> tuple[float, float]:
        total_unc = 0.0

        if self.counter > self.stack_layers and self.trained_once:
            normalized_state = normalize_observation(observation, self.statistics)
            data = torch.cat((normalized_state, actions), dim=1)
            # Get model uncertainty by sampling 100 parameters.
            self.world_model_1.load_state_dict(copy.deepcopy(self.world_model_0.state_dict()))
            sampler = SGLD_Sampler(self.world_model_1, step_size=0.00001, burn_in=0, num_steps=4, num_chains=3, tune=False)
            chains = sampler.sample_chains()
            # # Estimation from chains.
            preds = []
            aleatorics = []
            for i in range(len(chains)):
                for model_state_dict in chains[i].samples:
                    self.world_model_1.load_state_dict(model_state_dict)
                    pred, var_s = self.world_model_1(data)
                    aleatorics.append(var_s)
                    preds.append(pred)
            noises = torch.vstack(aleatorics).squeeze().detach().numpy()
            aleatoric = (noises ** 2).mean(axis=0) ** 0.5
            preds = torch.vstack(preds)
            all_means = preds.detach().numpy()
            epistemic = all_means.var(axis=0) ** 0.5
            aleatoric = np.minimum(aleatoric, 10e3)
            epistemic = np.minimum(epistemic, 10e3)
            total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
            total_unc = total_unc.mean()
        return total_unc, 0.0, preds



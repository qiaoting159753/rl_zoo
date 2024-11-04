from __future__ import division
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_zoo.agents.networks.world_models import World_Model
from rl_zoo.utils.helpers import denormalize_observation_delta, normalize_observation, normalize_observation_delta
from rl_zoo.agents.networks.world_models.bayesian.bayesian_sgld_classes import SGLD_Sampler
from torch.utils.data import DataLoader, TensorDataset


class CustomizedMLP(nn.Module):
    def __init__(self,
                 observation_size,
                 device,
                 num_actions,
                 hidden_sizes=None):
        super().__init__()
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
        log_prob = F.gaussian_nll_loss(mu, target=target, var=var_s)
        return {'log_prob': log_prob, 'MSE': mse.detach_()}


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
        self.world_model = CustomizedMLP(observation_size=observation_size,
                                         num_actions=num_actions,
                                         device=device,
                                         hidden_sizes=hidden_size)
        self.world_model_2 = CustomizedMLP(observation_size=observation_size,
                                           num_actions=num_actions,
                                           device=device,
                                           hidden_sizes=hidden_size)

        self.world_model.to(self.device)
        self.world_model_2.to(self.device)

        self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r)
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
        n_mean, n_var = self.world_model.forward(data)
        prediction = denormalize_observation_delta(n_mean, self.statistics)
        prediction += observation
        return prediction, None, n_mean, n_var

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train for On-policy flush training use.
        :param states:
        :param actions:
        :param next_states:
        """
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_state = normalize_observation(states, self.statistics)
        data = torch.cat((normalized_state, actions), dim=1)
        n_mean, n_var = self.world_model.forward(data)
        self.world_optimizers.zero_grad()
        model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
        model_loss.backward()
        self.world_optimizers.step()

        self.data[256 * (self.counter % self.stack_layers):256 * ((self.counter % self.stack_layers) + 1),
        :] = copy.deepcopy(data)
        self.target[256 * (self.counter % self.stack_layers):256 * ((self.counter % self.stack_layers) + 1),
        :] = copy.deepcopy(delta_targets_normalized)

        if self.counter > self.stack_layers:
            self.world_model_2.data = copy.deepcopy(self.data)
            self.world_model_2.data.to(self.device)
            self.world_model_2.target = copy.deepcopy(self.target)
            self.world_model_2.target.to(self.device)
            self.trained_once = True
            self.world_model_2.dataloader = DataLoader(TensorDataset(self.data, self.target), batch_size=50)
        self.counter += 1

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        total_unc = 0.0
        if self.counter > self.stack_layers:
            normalized_state = normalize_observation(observation, self.statistics)
            data = torch.cat((normalized_state, actions), dim=1)

            self.world_model_2.load_state_dict(copy.deepcopy(self.world_model.state_dict()))
            sampler = SGLD_Sampler(self.world_model_2,
                                   step_size=0.01,
                                   num_steps=10,
                                   num_chains=5,
                                   burn_in=3,
                                   pretrain=False,
                                   tune=False)
            chains = sampler.sample_chains()

            # # Estimation from chains.
            preds = []
            aleatorics = []
            for i in range(len(chains)):
                for model_state_dict in chains[i].samples:
                    self.world_model_2.load_state_dict(model_state_dict)
                    pred, var_s = self.world_model_2(data)
                    aleatorics.append(var_s)
                    preds.append(pred)
            # noises = torch.vstack(aleatorics).squeeze().detach().numpy()
            # aleatoric = (noises ** 2).mean(axis=0) ** 0.5
            # aleatoric = aleatoric.mean()
            preds = torch.vstack(preds)
            epistemic = torch.var(preds, dim=0).mean().item()
            # aleatoric = np.minimum(aleatoric, 10e3)
            # epistemic = np.minimum(epistemic, 10e3)
            # total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
            total_unc = epistemic
        return total_unc, 0.0

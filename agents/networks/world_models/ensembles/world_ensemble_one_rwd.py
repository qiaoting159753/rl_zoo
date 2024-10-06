import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim

from agents.networks.world_models.deterministic import Probabilistic_Dynamics, Probabilistic_SAS_Reward
from agents.networks.world_models.simple import Simple_SAS_Reward, Simple_NS_Reward
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta
from utils import denormalize_observation_delta, normalize_observation


def sig(x):
    """
    Sigmoid
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


class Ensemble_Dyna_One_Reward(World_Model):
    """
    World Model

    """
    def __init__(self, observation_size: int, num_actions: int, num_models: int, l_r: float, device: str,
                 boost_inter: int = 3, hidden_size: int = 256, sas: bool = True, prob_rwd: bool = False):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)
        self.sas = sas
        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions
        if not prob_rwd:
            if sas:
                self.reward_network = Simple_SAS_Reward(
                    observation_size=observation_size,
                    num_actions=num_actions,
                    hidden_size=hidden_size,
                )
            else:
                self.reward_network = Simple_NS_Reward(
                    observation_size=observation_size,
                    num_actions=num_actions,
                    hidden_size=hidden_size,
                )
        else:
            if sas:
                self.reward_network = Probabilistic_SAS_Reward(
                    observation_size=observation_size,
                    num_actions=num_actions,
                    hidden_size=hidden_size,
                )
        self.curr_losses = np.ones((self.num_models,)) * 5
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=l_r)
        self.world_models = [
            Probabilistic_Dynamics(
                observation_size=observation_size,
                num_actions=num_actions,
                hidden_size=hidden_size,
            )
            for _ in range(self.num_models)
        ]
        self.optimizers = [optim.Adam(self.world_models[i].parameters(), lr=l_r) for i in range(self.num_models)]
        self.statistics = {}
        # Bring all reward prediction and dynamic rediction networks to device.
        self.device = device
        self.reward_network.to(self.device)
        for model in self.world_models:
            model.to(device)
        self.boost_inter = boost_inter
        self.update_counter = 0

    def pred_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor):
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return:
        """
        if self.sas:
            pred_rewards = self.reward_network(observation, action, next_observation)
        else:
            pred_rewards = self.reward_network(next_observation, action)
        return pred_rewards, None, None

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        norm_means = []
        norm_vars = []
        normalized_observation = normalize_observation(observation, self.statistics)
        # Iterate over the neural networks and get the predictions
        for model in self.world_models:
            # Predict delta
            n_mean, n_var = model.forward(normalized_observation, actions)
            norm_means.append(n_mean)
            norm_vars.append(n_var)
        predictions_vars = torch.stack(norm_vars)
        predictions_norm_means = torch.stack(norm_means)
        # Normalized
        predictions_means = denormalize_observation_delta(predictions_norm_means, self.statistics)
        all_predictions = predictions_means + observation
        denorm_avg = torch.mean(predictions_means, dim=0)
        prediction = denorm_avg + observation
        # prediction = torch.mean(all_predictions, dim=0)
        return prediction, all_predictions, predictions_norm_means, predictions_vars

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        # This boosting part is useless, cause inaccuracy.
        # weights = 1.5 - sig(self.curr_losses)
        # weights /= np.max(weights)
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert (
                states.shape[1] + actions.shape[1]
                == self.num_actions + self.observation_size
        )
        # min_ = np.min(self.curr_losses)
        # max_ = np.max(self.curr_losses)
        # delta = max_ - min_
        # if delta == 0:
        #     delta = 0.1
        # temp = (self.curr_losses - min_) / delta * 5.0
        # temp = sig(temp)
        # temp[index] *
        index = int(math.floor(self.update_counter / self.boost_inter))
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_state = normalize_observation(states, self.statistics)
        n_mean, n_var = self.world_models[index].forward(normalized_state, actions)
        model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
        self.optimizers[index].zero_grad()
        model_loss.backward()
        self.optimizers[index].step()
        self.curr_losses[index] = model_loss.item()
        self.update_counter += 1
        self.update_counter %= self.boost_inter * self.num_models

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        assert len(next_states.shape) >= 2
        self.reward_optimizer.zero_grad()
        if self.sas:
            rwd_mean = self.reward_network(states, actions, next_states)
        else:
            rwd_mean = self.reward_network(next_states, actions)
        reward_loss = F.mse_loss(rwd_mean, rewards)
        reward_loss.backward()
        self.reward_optimizer.step()

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        """
        Estimate uncertainty.

        :param observation:
        :param actions:
        """
        means = []
        vars_s = []
        normalized_state = normalize_observation(observation, self.statistics)
        for model in self.world_models:
            mean, var = model.forward(normalized_state, actions)
            means.append(mean)
            vars_s.append(var)
        noises = torch.stack(vars_s).squeeze().detach().numpy()
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        all_means = torch.stack(means).squeeze().detach().numpy()
        epistemic = all_means.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc)
        # Reward Uncertainty
        sample_times = 20
        samples = []
        for i in range(len(means)):
            dist = torch.distributions.Normal(means[i], vars_s[i])
            samples.append(dist.sample([sample_times]))
        samples = torch.vstack(samples).squeeze()
        samples = denormalize_observation_delta(samples, self.statistics)
        samples += observation
        observationss = torch.repeat_interleave(observation, repeats=sample_times * self.num_models, dim=0)
        actionss = torch.repeat_interleave(actions, repeats=sample_times * self.num_models, dim=0)
        rewards, _, _ = self.pred_rewards(observationss, actionss, samples)
        var_reward = torch.var(rewards, dim=0).item()
        return uncert, var_reward

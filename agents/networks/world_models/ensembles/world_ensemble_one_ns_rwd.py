import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim

from agents.networks.world_models.deterministic import (
    Probabilistic_Dynamics,
)
from agents.networks.world_models.simple import (
    Simple_NS_Reward,
)
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, denormalize_observation_delta


def sig(x):
    """
    Sigmoid
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


class Ensemble_Dyna_One_NS_Reward(World_Model):
    """
    Spec
    """

    def __init__(self, observation_size: int, num_actions: int, num_models: int, l_r: float, device: str,
                 boost_inter: int = 3, hidden_size: int = 128):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)

        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.curr_losses = np.ones((self.num_models,)) * 5

        self.reward_network = Simple_NS_Reward(
            observation_size=observation_size,
            num_actions=num_actions,
            hidden_size=hidden_size,
        )
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=l_r)

        self.models = [
            Probabilistic_Dynamics(
                observation_size=observation_size,
                num_actions=num_actions,
                hidden_size=hidden_size,
            )
            for _ in range(self.num_models)
        ]

        self.optimizers = [optim.Adam(self.models[i].parameters(), lr=l_r) for i in range(self.num_models)]

        self.statistics = {}

        # Bring all reward prediction and dynamic rediction networks to device.
        self.device = device
        self.reward_network.to(self.device)
        for model in self.models:
            model.to(device)

        self.boost_inter = boost_inter
        self.update_counter = 0

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
        for model in self.models:
            model.statistics = statistics

    def pred_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor):
        pred_rewards = self.reward_network(observation)
        return pred_rewards, None, None

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        means = []
        norm_means = []
        norm_vars = []

        # This boosting part is useless, cause inaccuracy.
        # weights = 1.5 - sig(self.curr_losses)
        # weights /= np.max(weights)

        # Iterate over the neural networks and get the predictions
        for counter in range(self.num_models):
            # Predict delta
            mean, n_mean, n_var = self.models[counter].forward(observation, actions)
            # mean *= weights[counter]
            means.append(mean)
            norm_means.append(n_mean)
            norm_vars.append(n_var)
        # Normalized
        predictions_means = torch.stack(means)
        predictions_norm_means = torch.stack(norm_means)
        predictions_vars = torch.stack(norm_vars)

        # Get rid of the nans
        # not_nans = []
        # for i in range(self.num_models):
        #     if not torch.any(torch.isnan(predictions_means[i])):
        #         not_nans.append(i)
        # if len(not_nans) == 0:
        #     logging.info("Predicting all Nans")
        #     sys.exit()
        # Random Take next state.
        # rand_ind = random.randint(0, len(not_nans) - 1)
        # prediction = predictions_means[not_nans[rand_ind]]
        # next = current + delta
        prediction = torch.mean(predictions_means, dim=0)
        prediction += observation

        all_predictions = torch.stack(means)
        for j in range(all_predictions.shape[0]):
            all_predictions[j] += observation

        return prediction, all_predictions, predictions_norm_means, predictions_vars

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:

        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert (
                states.shape[1] + actions.shape[1]
                == self.num_actions + self.observation_size
        )

        min_ = np.min(self.curr_losses)
        max_ = np.max(self.curr_losses)
        delta = max_ - min_
        if delta == 0:
            delta = 0.1
        temp = (self.curr_losses - min_) / delta * 5.0
        temp = sig(temp)

        index = int(math.floor(self.update_counter / self.boost_inter))
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        _, n_mean, n_var = self.models[index].forward(states, actions)
        model_loss = temp[index] * F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
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
        # assert len(actions.shape) == 2
        # assert (
        #         next_states.shape[1] + actions.shape[1]
        #         == self.num_actions + self.observation_size
        # )
        self.reward_optimizer.zero_grad()
        rwd_mean = self.reward_network.forward(next_states)
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
        # sample_times = 100
        # _, _, mean, var = self.pred_next_states(observation, actions)
        # # # Sample next state several times, and estimate reward uncertianty.
        # sample1 = torch.distributions.Normal(mean, var).sample([sample_times])
        # sample1 = sample1.squeeze()
        # tempa = []
        # for lm in range(self.num_models):
        #     tempa.append(sample1[:, lm, :])
        # sample1 = torch.vstack(tempa)
        # sample1i = denormalize_observation_delta(sample1, self.statistics)
        # sample1i += observation
        # dyna_uncert = torch.mean(torch.var(sample1i, dim=0)).item()
        # # multi_observation = torch.repeat_interleave(observation, self.num_models * sample_times, dim=0)
        # # multi_reward = torch.repeat_interleave(actions, self.num_models * sample_times, dim=0)
        # # reward, _, _ = self.pred_rewards(multi_observation, multi_reward, sample1i)
        # # rwd_uncert = torch.var(reward).item()

        means = []
        vars = []
        for model in self.models:
            _, mean, var = model.forward(observation, actions)
            means.append(mean)
            vars.append(var)
        all_vars = torch.stack(vars).squeeze().detach().numpy()
        all_means = torch.stack(means).squeeze().detach().numpy()
        noises = all_vars
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        epistemic = all_means.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc)
        return uncert, 0.0

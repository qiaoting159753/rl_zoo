import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim
from rl_zoo.utils import normalize_observation
from rl_zoo.agents.networks.world_models.deterministic import (
    NVP_Flows,
)
from rl_zoo.agents.networks.world_models.simple import (
    Simple_SAS_Reward,
)
from rl_zoo.agents.networks.world_models import World_Model
from rl_zoo.utils.helpers import normalize_observation_delta


class Ensemble_NF_One_SAS_Reward(World_Model):
    """
    World Model

    """

    def __init__(self, observation_size: int, num_actions: int, num_models: int, l_r: float, device: str,
                 hidden_size: int = 128):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)
        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.reward_network = Simple_SAS_Reward(
            observation_size=observation_size,
            num_actions=num_actions,
            hidden_size=hidden_size,
        )
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=l_r)

        self.models = [NVP_Flows(state_dim=self.observation_size, act_dim=self.num_actions) for _ in
                       range(self.num_models)]

        self.optimizers = [optim.Adam(self.models[i].parameters(), lr=l_r) for i in range(self.num_models)]

        self.statistics = {}

        # Bring all reward prediction and dynamic rediction networks to device.
        self.device = device
        self.reward_network.to(self.device)
        for model in self.models:
            model.to(device)

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
        # For each model, train with different data.
        mini_batch_size = int(math.floor(states.shape[0] / self.num_models))
        for i in range(self.num_models):
            sub_states = states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_actions = actions[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_next_states = next_states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_target = sub_next_states - sub_states
            delta_targets_normalized = normalize_observation_delta(sub_target, self.statistics)
            s_n_a = torch.cat((delta_targets_normalized, sub_actions), dim=1)

            _, z_, log_dets = self.models[i].forward(sub_states, sub_actions)
            # Reverse KLD: Log_q - Log_p = (Log_q0 - forward_log_det) - (-MSE)
            mse_loss = F.mse_loss(z_, s_n_a, reduction="sum")
            model_loss = torch.mean(-1 * log_dets + mse_loss)

            self.optimizers[i].zero_grad()
            model_loss.backward()
            self.optimizers[i].step()

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        means = []
        # Iterate over the neural networks and get the predictions
        for model in self.models:
            # Predict delta
            mean, _, _ = model.forward(observation, actions)
            means.append(mean)
        # Normalized
        predictions_means = torch.stack(means)
        # Get rid of the nans
        not_nans = [i for i in range(self.num_models)]
        # Random Take next state.
        rand_ind = random.randint(0, len(not_nans) - 1)
        prediction = predictions_means[not_nans[rand_ind]]
        # next = current + delta
        prediction += observation
        all_predictions = torch.stack(means)
        for j in range(all_predictions.shape[0]):
            all_predictions[j] += observation
        return prediction, all_predictions, prediction, prediction

    def pred_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor):
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return:
        """
        pred_rewards = self.reward_network(observation, action, next_observation)
        return pred_rewards, None, None

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        assert len(next_states.shape) >= 2
        self.reward_optimizer.zero_grad()
        rwd_mean = self.reward_network(states, actions, next_states)
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
        normalized_obs = normalize_observation(observation, self.statistics)
        gt_s_a = torch.cat((normalized_obs, actions), dim=1)
        # reversed_results = []
        # mse_losses = 0.0
        # for model in self.models:
        #     _, z, _ = model.forward(observation, actions)
        #     z = z.detach()
        #     z_, _ = model.reverse(z)
        #     mse_loss = F.mse_loss(z_, gt_s_a).item()
        #     mse_losses += mse_loss
        #     z_ = z_.detach()
        #     reversed_results.append(z_)
        # reversed_results = torch.vstack(reversed_results)
        # dyna_uncert = torch.mean(torch.var(reversed_results, dim=0))
        # rwd_uncert = 0.0
        means = []
        vars = []
        for model in self.models:
            _, mean, var = model.forward(observation, actions)
            means.append(mean)
            vars.append(var)
        all_vars = torch.stack(vars)
        all_means = torch.stack(means)

        print(all_vars.shape)
        print(all_means.shape)

        noises = all_vars
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        epistemic = all_means.var(dim=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

        return 0.0, 0.0

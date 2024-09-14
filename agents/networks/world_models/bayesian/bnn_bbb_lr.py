import torch
import numpy as np
from agents.networks.world_models import World_Model
from agents.networks.world_models.bayesian.bayesian_javirantoran_bbb_lr import bayes_linear_LR_2L

from utils import normalize_observation_delta
import torch.nn.functional as F
from torch import optim
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
from utils import normalize_observation, denormalize_observation_delta


class Bayesian_World_Model_BBB_LR(World_Model):
    def __init__(self, observation_size: int, num_actions: int, l_r: float, device: str, hidden_size: int = 128):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)
        self.statistics = None
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.l_r = l_r
        self.device = device
        self.world_model = bayes_linear_LR_2L(observation_size + num_actions, 2 * observation_size, hidden_size,
                                              prior_sig=0.1)
        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)
        self.reward_optimizers = optim.Adam(self.reward_model.parameters(), lr=l_r)
        self.world_optimizers = optim.Adam(self.world_model.parameters(), lr=0.0001)
        self.reward_model.to(self.device)
        self.world_model.to(self.device)

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
        # self.world_model.statistics = statistics

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train the dynamic of world model.
        :param states:
        :param actions:
        :param next_states:
        """
        samples = 10
        target = next_states - states
        y = normalize_observation_delta(target, self.statistics)
        normalized_obs = normalize_observation(states, self.statistics)
        x = torch.cat((normalized_obs, actions), dim=1)

        # x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
        self.world_optimizers.zero_grad()
        mlpdw_cum = 0
        Edkl_cum = 0
        for i in range(samples):
            out, tlqw, tlpw = self.world_model(x, sample=True)
            Edkl_i = (tlqw - tlpw)

            mean_pred = out[:, :self.observation_size]
            var_pred = out[:, self.observation_size:]
            var_pred = torch.tanh(var_pred)
            var_pred = torch.exp(var_pred)
            mlpdw_i = F.gaussian_nll_loss(input=mean_pred, target=y, var=var_pred).mean()

            # mlpdw_i = F.mse_loss(out, y).mean()

            mlpdw_cum += mlpdw_i
            Edkl_cum = Edkl_cum + Edkl_i
        mlpdw = mlpdw_cum / samples
        Edkl = Edkl_cum / (samples * 20000)
        loss = Edkl + mlpdw
        loss.backward()
        self.world_optimizers.step()

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        """
        Estimate next state uncertainty and reward uncertainty.

        :param observation:
        :param actions:
        :return:
        """
        # logging.info("Not Implemented")
        sample = 10
        preds = []
        normalized_obs = normalize_observation(observation, self.statistics)
        x = torch.cat((normalized_obs, actions), dim=1)
        for _ in range(sample):
            pred, _, _ = self.world_model(x, sample=True)

            mean_pred = pred[:, :self.observation_size]
            var_pred = pred[:, self.observation_size:]
            var_pred = torch.tanh(var_pred)
            var_pred = torch.exp(var_pred)
            sample1 = torch.distributions.Normal(mean_pred, var_pred).sample([sample])
            preds.append(sample1)

            # preds.append(pred)

        preds = torch.vstack(preds).squeeze()
        mean_deltas = denormalize_observation_delta(preds, self.statistics)
        preds = mean_deltas + observation
        dyna_var = torch.sum(torch.var(preds, dim=0)).item()
        return dyna_var, 0.0

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        """
        Train the reward prediction with or without world model dynamics.

        :param states:
        :param actions:
        :param next_states:
        :param rewards:
        """
        self.reward_optimizers.zero_grad()
        rwd_mean, rwd_var = self.reward_model.forward(states, actions, next_states)
        reward_loss = F.gaussian_nll_loss(input=rwd_mean, target=rewards, var=rwd_var).mean()
        reward_loss.backward()
        self.reward_optimizers.step()

    def pred_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor
                     ):
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return:
        """
        pred_reward, reward_var = self.reward_model.forward(observation, action, next_observation)
        return pred_reward, None, reward_var

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param observation:
        :param actions:
        """
        sample = 3
        preds = []
        normalized_obs = normalize_observation(observation, self.statistics)
        x = torch.cat((normalized_obs, actions), dim=1)
        for i in range(sample):
            pred, _, _ = self.world_model(x, sample=True)
            mean_pred = pred[:, :self.observation_size]
            var_pred = pred[:, self.observation_size:]
            var_pred = torch.tanh(var_pred)
            var_pred = torch.exp(var_pred)
            sample1 = torch.distributions.Normal(mean_pred, var_pred).sample([sample])
            preds.append(sample1)

            # preds.append(pred)

        preds = torch.vstack(preds).squeeze()
        mean_deltas = denormalize_observation_delta(preds, self.statistics)
        preds = mean_deltas + observation
        preds = torch.mean(preds, dim=0).unsqueeze(dim=0)
        return preds, None, None, None

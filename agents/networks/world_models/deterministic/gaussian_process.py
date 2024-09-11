import torch
import numpy as np
from agents.networks.world_models import World_Model
from utils import normalize_observation_delta
import torch.nn.functional as F
from torch import optim
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
from utils import normalize_observation, denormalize_observation_delta
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

class Gaussian_Process():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel = gp.kernels.RBF(
            input_dim=input_dim, variance=torch.tensor(6.0), lengthscale=torch.tensor(0.05)
        )
        x = torch.zeros((1,self.input_dim))
        y = torch.zeros((1,self.output_dim))
        self.gpr = gp.models.GPRegression(x, y, self.kernel, noise=torch.tensor(0.2))

    def fit(self, x, y):
        self.gpr.set_data(x, y)
        optimizer = torch.optim.Adam(self.gpr.parameters(), lr=0.001)
        gp.util.train(self.gpr, optimizer)


class Gaussian_Process_World_Model(World_Model):
    def __init__(self, observation_size: int, num_actions: int, l_r: float, device: str, hidden_size: int = 128):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)
        self.statistics = None
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.l_r = l_r
        self.device = device

        self.world_model = Gaussian_Process(self.observation_size + self.num_actions, self.observation_size)

        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)

        self.reward_optimizers = optim.Adam(self.reward_model.parameters(), lr=l_r)
        # self.world_optimizers = optim.Adam(self.world_model.parameters(), lr=0.0001)
        self.reward_model.to(self.device)
        # self.world_model.to(self.device)

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

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        """
        Estimate next state uncertainty and reward uncertainty.

        :param observation:
        :param actions:
        :return:
        """
        return 0.0, 0.0

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


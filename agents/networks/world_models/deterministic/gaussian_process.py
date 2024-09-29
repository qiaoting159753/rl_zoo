import torch
import numpy as np
from agents.networks.world_models import World_Model
import torch.nn.functional as F
from torch import optim
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
import pyro
import pyro.contrib.gp as gp


class Gaussian_Process_World_Model(World_Model):
    def __init__(self, observation_size: int, num_actions: int, device: str,
                 l_r: float,
                 noise: float = 0.2,
                 hidden_size: int = 128,
                 train_iter: int = 10):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)
        self.gpr = None
        self.statistics = None
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.l_r = l_r
        self.device = device
        self.noise = noise
        self.train_iter = train_iter
        self.kernel = gp.kernels.RBF(input_dim=observation_size + num_actions)
        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)
        self.reward_optimizers = optim.Adam(self.reward_model.parameters(), lr=l_r)
        self.reward_model.to(self.device)

    def set_statistics(self, statistics: dict) -> None:
        """
        Update all statistics for normalization for all world models and the
        ensemble itself.

        :param (Dictionary) statistics:
        """
        return

    def train_world(
            self,
            states: list,
            actions: list,
            next_states: list,
    ) -> None:
        """
        For nothing.
        :param states:
        :param actions:
        :param next_states:
        :return:
        """
        return

    def train_world_all(
            self,
            states: list,
            actions: list,
            next_states: list,
    ) -> None:
        """
        Train the dynamic of world model.
        :param states:
        :param actions:
        :param next_states:
        """
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        next_states = torch.FloatTensor(np.array(next_states))
        y_y = next_states.T
        x_x = torch.cat((states, actions), dim=1)
        gpr = gp.models.GPRegression(x_x, y_y, self.kernel, noise=torch.tensor(self.noise))
        optimizer = torch.optim.Adam(gpr.parameters(), lr=self.l_r)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        for _ in range(self.train_iter):
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
        self.gpr = gpr

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param observation:
        :param actions:
        """
        # sample = 3
        # preds = []
        x = torch.cat((observation, actions), dim=1)
        preds, _ = self.gpr(x, full_cov=True)
        preds = preds.T
        preds += observation
        # preds = torch.mean(preds, dim=0).unsqueeze(dim=0)
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
        x = torch.cat((observation, actions), dim=1)
        _, covs = self.gpr(x, full_cov=True)
        uncet = torch.sum(torch.squeeze(covs)).detach().cpu().numpy()
        return uncet, 0.0

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

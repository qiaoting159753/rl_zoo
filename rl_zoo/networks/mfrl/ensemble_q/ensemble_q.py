import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Ensemble_Probabilistic_Critic(nn.Module):
    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            num_ensembles: int = 5,
            hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]
        self.num_ensembles = num_ensembles
        self.hidden_size = hidden_size
        self.q_list = []
        for _ in range(num_ensembles):
            q_n = nn.Sequential(
                nn.Linear(observation_size + num_actions, self.hidden_size[0]),
                nn.ReLU(),
                nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                nn.ReLU(),
                nn.Linear(self.hidden_size[1], 2),
            )
            self.q_list.append(q_n)

    def forward(self, state: torch.Tensor, action: torch.Tensor, index: int):
        obs_action = torch.cat([state, action], dim=1)
        q_ab = self.q_list[index](obs_action)
        q_a = q_ab[:, 0]  # Mean
        q_b = q_ab[:, 1]  # Var
        q_b = torch.tanh(q_b)
        q_b = torch.exp(q_b)
        # Use: F.gaussian_nll_loss(input = q_a, target=target_q, var=q_b)
        # Not use: F.mse_loss(input=q_a, target=target_q)
        return q_a, q_b

    def forward_all(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q_s_mean = []
        q_s_vars = []
        for i in range(self.num_ensembles):
            q_ab = self.q_list[i](obs_action)
            q_a = q_ab[:, 0]
            q_b = q_ab[:, 1]
            q_b = torch.tanh(q_b)
            q_b = torch.exp(q_b)
            q_s_mean.append(q_a)
            q_s_vars.append(q_b)
        return torch.stack(q_s_mean), torch.stack(q_s_vars)

    def estimate_uncertainty(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, var_s = self.forward_all(state, action)
        noises = torch.stack(var_s).cpu().squeeze().detach().numpy()
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        all_means = means.cpu().squeeze().detach().numpy()
        epistemic = all_means.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc)
        return uncert


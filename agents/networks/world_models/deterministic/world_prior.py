from __future__ import division
import torch
import torch.nn.functional as F
from torch import nn
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, denormalize_observation_delta, normalize_observation


class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = F.softplus(logv) + 0.0001
        alpha = F.softplus(logalpha) + 1.0001
        beta = F.softplus(logbeta) + 0.0001
        return mu, v, alpha, beta

# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ...
def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()

def evidential_regression(dist_params, y, lamb=1.0):
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)


class Prior_World_Model(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r=0.0001,
                 hidden_size=128,
                 device='cpu',
                 sas: bool = True,
                 prob_rwd:bool = False):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)

        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model = nn.Sequential(
            nn.Linear(self.observation_size + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            NormalInvGamma(hidden_size, self.observation_size),
        )
        self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r)
        self.world_model.to(self.device)

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train for On-policy flush training use.
        """
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_state = normalize_observation(states, self.statistics)
        s_n_a = torch.cat((normalized_state, actions), dim=1)
        pred = self.world_model.forward(s_n_a)
        loss = evidential_regression(pred, delta_targets_normalized, lamb=1e-1)
        self.world_optimizers.zero_grad()
        loss.backward()
        self.world_optimizers.step()

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_state = normalize_observation(observation, self.statistics)
        s_n_a = torch.cat((normalized_state, actions), dim=1)
        pred = self.world_model(s_n_a)
        mu, v, alpha, beta = (d.squeeze() for d in pred)
        prediction = denormalize_observation_delta(mu.unsqueeze(dim=0), self.statistics)
        prediction += observation
        return prediction, None, None, None

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        normalized_state = normalize_observation(observation, self.statistics)
        s_n_a = torch.cat((normalized_state, actions), dim=1)
        with torch.no_grad():
            pred = self.world_model(s_n_a)
        mu, v, alpha, beta = (d.squeeze() for d in pred)
        var = torch.sqrt(beta / (v * (alpha - 1)))
        uncert = torch.mean(var).item()

        # Reward Uncertainty
        sample_times = 100
        dist = torch.distributions.Normal(mu, var)
        samples = (dist.sample([sample_times]))
        samples = samples.squeeze()
        samples = denormalize_observation_delta(samples, self.statistics)
        samples += observation
        observationss = torch.repeat_interleave(observation, repeats=sample_times, dim=0)
        actionss = torch.repeat_interleave(actions, repeats=sample_times, dim=0)
        if self.sas:
            if self.prob_rwd:
                rewards, rwd_var = self.reward_network(observationss, actionss, samples)
                epis_uncert = torch.var(rewards, dim=0).item()
                uncert_rwd = epis_uncert + rwd_var
            else:
                rewards = self.reward_network(observationss, actionss, samples)
                uncert_rwd = torch.var(rewards, dim=0).item()
        else:
            if self.prob_rwd:
                rewards, rwd_var = self.reward_network(samples, actionss)
                epis_uncert = torch.var(rewards, dim=0).item()
                uncert_rwd = epis_uncert + rwd_var
            else:
                rewards = self.reward_network(samples, actionss)
                uncert_rwd = torch.var(rewards, dim=0).item()
        return uncert, uncert_rwd

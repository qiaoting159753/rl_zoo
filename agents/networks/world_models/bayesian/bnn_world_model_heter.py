import torch
import torch.nn as nn
import numpy as np
from agents.networks.world_models import World_Model
from torch.autograd import Variable
from utils import normalize_observation_delta
import torch.nn.functional as F
from torch import optim
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
from utils import normalize_observation, denormalize_observation_delta


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5 * (weights - self.mu) ** 2 / self.sigma ** 2
        log_coeff = -0.5 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))
        return (exponent + log_coeff).sum()


class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        self.weight_mus = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.weight_rhos = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -3))
        self.bias_mus = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -3))

    def forward(self, x):
        # sample gaussian noise for each weight and each bias
        weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
        bias_epsilons = Variable(self.bias_mus.data.new(self.bias_mus.size()).normal_())
        # calculate the weight and bias stds from the rho parameters
        weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
        bias_stds = torch.log(1 + torch.exp(self.bias_rhos))
        # calculate samples from the posterior from the sampled noise and mus/stds
        weight_sample = self.weight_mus + weight_epsilons * weight_stds
        bias_sample = self.bias_mus + bias_epsilons * bias_stds
        output = torch.mm(x, weight_sample) + bias_sample
        # computing the KL loss term
        prior_cov, varpost_cov = self.prior.sigma ** 2, weight_stds ** 2
        KL_loss = 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * weight_stds.numel()
        KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        KL_loss = KL_loss + 0.5 * ((self.weight_mus - self.prior.mu) ** 2 / prior_cov).sum()
        prior_cov, varpost_cov = self.prior.sigma ** 2, bias_stds ** 2
        KL_loss = KL_loss + 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * bias_stds.numel()
        KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        KL_loss = KL_loss + 0.5 * ((self.bias_mus - self.prior.mu) ** 2 / prior_cov).sum()
        return output, KL_loss


class BBP_Heteroscedastic_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(BBP_Heteroscedastic_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, num_units, gaussian(0, 1))
        self.layer2 = BayesLinear_Normalq(num_units, num_units, gaussian(0, 1))
        self.layer3 = BayesLinear_Normalq(num_units, output_dim, gaussian(0, 1))

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)

        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        x, KL_loss = self.layer3(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)

        return x, KL_loss_total


class Bayesian_World_Model_2(World_Model):
    def __init__(self, observation_size: int, num_actions: int, l_r: float, device: str, hidden_size: int = 128):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size)
        self.statistics = None
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.l_r = l_r
        self.device = device
        self.world_model = BBP_Heteroscedastic_Model(observation_size + num_actions, observation_size, hidden_size)

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
            out, KL_loss_total = self.world_model(x)

            # mean_pred = out[:, :self.observation_size]
            # var_pred = out[:, self.observation_size:]
            # var_pred = torch.tanh(var_pred)
            # var_pred = torch.exp(var_pred)
            # mlpdw_i = F.gaussian_nll_loss(input=mean_pred, target=y, var=var_pred).mean()

            mlpdw_i = F.mse_loss(out, y).mean()
            mlpdw_cum += mlpdw_i
            Edkl_cum += KL_loss_total

        Edkl_cum /= (samples * states.shape[0])
        loss = mlpdw_cum + Edkl_cum
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
        dyna_var = 0.0
        # logging.info("Not Implemented")
        # sample = 10
        # preds = []
        # normalized_obs = normalize_observation(observation, self.statistics)
        # x = torch.cat((normalized_obs, actions), dim=1)
        # for _ in range(sample):
        #     pred, _ = self.world_model(x)
        #
        #     # mean_pred = pred[:, :self.observation_size]
        #     # var_pred = pred[:, self.observation_size:]
        #     # var_pred = torch.tanh(var_pred)
        #     # var_pred = torch.exp(var_pred)
        #     # sample1 = torch.distributions.Normal(mean_pred, var_pred).sample([sample])
        #     # preds.append(sample1)
        #
        #     preds.append(pred)
        #
        # preds = torch.vstack(preds).squeeze()
        # mean_deltas = denormalize_observation_delta(preds, self.statistics)
        # preds = mean_deltas + observation
        # dyna_var = torch.sum(torch.var(preds, dim=0)).item()
        # # prediction = torch.mean(preds, dim=0).unsqueeze(dim=0)
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
        sample = 10
        preds = []
        normalized_obs = normalize_observation(observation, self.statistics)
        x = torch.cat((normalized_obs, actions), dim=1)
        for i in range(sample):
            pred, _ = self.world_model(x)
            # mean_pred = pred[:, :self.observation_size]
            # var_pred = pred[:, self.observation_size:]
            # var_pred = torch.tanh(var_pred)
            # var_pred = torch.exp(var_pred)
            # sample1 = torch.distributions.Normal(mean_pred, var_pred).sample([sample])
            # preds.append(sample1)
            preds.append(pred)
        preds = torch.vstack(preds).squeeze()
        mean_deltas = denormalize_observation_delta(preds, self.statistics)
        preds = mean_deltas + observation
        preds = torch.mean(preds, dim=0).unsqueeze(dim=0)
        return preds, None, None, None


import torch
import numpy as np
from agents.networks.world_models import World_Model
from agents.networks.world_models.bayesian.bnn_bbb_lr import bayes_linear_lr
from agents.networks.world_models.bayesian.bnn_bbb_vi import bayes_linear_vi
from agents.networks.world_models.bayesian.z_hyper_bnn_bbb import Hyper_BBP_Heteroscedastic_Model
from utils import normalize_observation_delta
import torch.nn.functional as F
from torch import optim
from utils import normalize_observation, denormalize_observation_delta


class Bayesian_World_Model_BBB(World_Model):
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 l_r: float,
                 device: str,
                 sigma: float,
                 ratio: float,
                 hidden_size: int = 256,
                 option: int = 1,
                 sas: bool = True,
                 prob_rwd: bool = False):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        self.statistics = None
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.l_r = l_r
        self.ratio = ratio
        self.device = device
        self.sas = sas
        self.prob_rwd = prob_rwd
        if option == 0:
            self.world_model = Hyper_BBP_Heteroscedastic_Model(observation_size + num_actions, 2 * observation_size,
                                                               hidden_size)
        if option == 1:
            self.world_model = bayes_linear_vi(observation_size + num_actions, 2 * observation_size, hidden_size, sigma)
        if option == 2:
            self.world_model = bayes_linear_lr(observation_size + num_actions, 2 * observation_size, hidden_size, sigma)

        self.inv_total_params = 1.0 / (sum(p.numel() for p in self.world_model.parameters()))
        self.world_optimizers = optim.Adam(self.world_model.parameters(), lr=self.l_r)
        self.world_model.to(self.device)

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
        normalized_target = normalize_observation_delta(target, self.statistics)
        normlized_state = normalize_observation(states, self.statistics)
        x = torch.cat((normlized_state, actions), dim=1)
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
            mlpdw_i = F.gaussian_nll_loss(input=mean_pred, target=normalized_target, var=var_pred).mean()
            # mlpdw_i = F.mse_loss(out, y).mean()
            mlpdw_cum += mlpdw_i
            Edkl_cum = Edkl_cum + Edkl_i
        mlpdw = mlpdw_cum / samples
        Edkl = (Edkl_cum / samples) * self.inv_total_params * self.ratio
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
        normlized_state = normalize_observation(observation, self.statistics)
        x = torch.cat((normlized_state, actions), dim=1)
        means = []
        vars = []
        sample_times = 5
        # for _ in range(sample_times):
        for i in range(sample_times):
            pred, _, _ = self.world_model.sample_predict(x, Nsamples=sample_times)
            pred = pred.squeeze()
            mean_pred = pred[:, :self.observation_size]
            var_pred = pred[:, self.observation_size:]
            var_pred = torch.tanh(var_pred)
            var_pred = torch.exp(var_pred)
            means.append(mean_pred)
            vars.append(var_pred)
        all_vars = torch.vstack(vars).squeeze().detach().cpu().numpy()
        all_means = torch.vstack(means).squeeze().detach().cpu().numpy()
        noises = all_vars ** 0.5
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        epistemic = all_means.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc).item()

        # Reward Uncertainty
        # sample_times = 100
        # dist = torch.distributions.Normal(mu, var)
        # samples = (dist.sample([sample_times]))
        # samples = samples.squeeze()
        # samples = denormalize_observation_delta(samples, self.statistics)
        # samples += observation
        # observationss = torch.repeat_interleave(observation, repeats=sample_times, dim=0)
        # actionss = torch.repeat_interleave(actions, repeats=sample_times, dim=0)
        # if self.sas:
        #     if self.prob_rwd:
        #         rewards, rwd_var = self.reward_network(observationss, actionss, samples)
        #         epis_uncert = torch.var(rewards, dim=0).item()
        #         uncert_rwd = epis_uncert + rwd_var
        #     else:
        #         rewards = self.reward_network(observationss, actionss, samples)
        #         uncert_rwd = torch.var(rewards, dim=0).item()
        # else:
        #     if self.prob_rwd:
        #         rewards, rwd_var = self.reward_network(samples, actionss)
        #         epis_uncert = torch.var(rewards, dim=0).item()
        #         uncert_rwd = epis_uncert + rwd_var
        #     else:
        #         rewards = self.reward_network(samples, actionss)
        #         uncert_rwd = torch.var(rewards, dim=0).item()
        uncert_rwd = 0.0
        return uncert, uncert_rwd

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normlized_state = normalize_observation(observation, self.statistics)
        normalized_obs_a = torch.cat((normlized_state, actions), dim=1)
        # for i in range(sample):
        #     pred, _, _ = self.world_model(normalized_obs_a)
        #     mean_pred = pred[:, :self.observation_size]
        #     var_pred = pred[:, self.observation_size:]
        #     var_pred = torch.tanh(var_pred)
        #     var_pred = torch.exp(var_pred)
        #     sample1 = torch.distributions.Normal(mean_pred, var_pred).sample([sample])
        #     preds.append(sample1)
        # preds = torch.vstack(preds).squeeze()
        pred, _, _ = self.world_model(normalized_obs_a, sample=False)
        preds = pred[:, :self.observation_size]
        mean_deltas = denormalize_observation_delta(preds, self.statistics)
        preds = mean_deltas + observation
        # preds = torch.mean(preds, dim=0).unsqueeze(dim=0)
        return preds, None, None, None

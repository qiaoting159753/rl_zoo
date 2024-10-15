from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, denormalize_observation_delta, normalize_observation
from .bayesian_aleximmer_la import KronLaplace
from utils.common import MLP


class Bayesian_World_Model_Laplace_AX(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r,
                 hidden_size,
                 sigma,
                 temperature,
                 prior_precision,
                 device,
                 sas,
                 prob_rwd,
                 train_both=False):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        self.sas = sas
        self.prob_rwd = prob_rwd
        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model = MLP(input_size=(observation_size + num_actions), output_size=2 * observation_size,
                               hidden_sizes=[128, 128])

        self.bnn = KronLaplace(model=self.world_model, likelihood="regression", sigma_noise=sigma,
                               temperature=temperature, prior_precision=prior_precision)

        self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r)
        self.world_model.to(self.device)

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_states = normalize_observation(observation, self.statistics)
        s_n_a = torch.cat((normalized_states, actions), dim=1)
        pred_s = self.world_model(s_n_a)
        n_mean_delta = pred_s[:, :self.observation_size]
        prediction = denormalize_observation_delta(n_mean_delta, self.statistics)
        prediction += observation
        return prediction, None, None, None

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train all
        :param states:
        :param actions:
        :param next_states:
        """
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_states = normalize_observation(states, self.statistics)
        s_n_a = torch.cat((normalized_states, actions), dim=1)
        pred_s = self.world_model(s_n_a)
        n_mean_delta = pred_s[:, :self.observation_size]
        n_log_delta = pred_s[:, self.observation_size:]
        var_s = torch.tanh(n_log_delta)
        n_log_delta = torch.exp(var_s)
        model_loss = F.gaussian_nll_loss(input=n_mean_delta, target=delta_targets_normalized, var=n_log_delta).mean()
        self.world_optimizers.zero_grad()
        model_loss.backward()
        self.world_optimizers.step()

        pred_s = self.world_model(s_n_a)
        x = pred_s[:, :self.observation_size]
        y_x = ((delta_targets_normalized - x) ** 2).detach()
        self.bnn.fit((s_n_a, torch.cat((delta_targets_normalized, y_x), dim=1)))
        self.bnn.optimize_prior_precision(pred_type="glm")

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        uncert = 0.0
        uncert_rwd = 0.0
        if self.bnn.n_data > 0:
            normalized_states = normalize_observation(observation, self.statistics)
            s_n_a = torch.cat((normalized_states, actions), dim=1)
            pred = self.bnn(s_n_a, pred_type="glm", link_approx="mc")
            mean, var = pred
            mean_np = mean.detach().cpu().numpy()
            var_all = torch.diagonal(var.squeeze()).unsqueeze(dim=0).detach().cpu().numpy()
            # Aleatoric: Last part of the mean
            # Epistemic: First/Second half of the var.
            epistemic = var_all[:, :self.observation_size]
            noises = mean_np[:, self.observation_size:]
            aleatoric = (noises ** 2).mean(axis=0) ** 0.5
            # epistemic = all_means.var(axis=0) ** 0.5
            aleatoric = np.minimum(aleatoric, 10e3)
            epistemic = np.minimum(epistemic, 10e3)
            total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
            uncert = np.mean(total_unc).item()

            if self.train_both:
                nn_s_a = torch.cat((observation, actions), dim=1)
                rrwd_pred = self.bnn_all(nn_s_a, pred_type="glm", link_approx="mc")
                rrwd_mean, rrwd_var = rrwd_pred
                rrwd_mean = rrwd_mean.squeeze().detach().cpu().numpy()
                rrwd_var = torch.diagonal(rrwd_var).squeeze().detach().cpu().numpy()
                epistemic = rrwd_var[0]
                if epistemic < 0.001: epistemic = 0.001
                aleatoric = rrwd_mean[1]
                if aleatoric < 0.001: aleatoric = 0.001
                uncert_rwd = (epistemic ** 2 + aleatoric ** 2) ** 0.5
            else:
                # # Sampling does not working well.
                var_s = torch.diagonal(var.squeeze()).unsqueeze(0)
                var_s[var_s < 0.00001] = 0.00001
                # Solution 1: Templete.
                sample_times = 10
                mean_dist = torch.distributions.Normal(mean, var_s)
                a = mean_dist.sample([sample_times]).squeeze()
                samples = []
                for i in range(a.shape[0]):
                    b_mean = a[i, :self.observation_size]
                    b_var = a[i, self.observation_size:]
                    b_var[b_var < 0.00001] = 0.00001
                    dist2 = torch.distributions.Normal(b_mean, b_var)
                    b = dist2.sample([sample_times])
                    samples.append(b.squeeze())
                samples = torch.vstack(samples)
                observationss = torch.repeat_interleave(observation, repeats=sample_times ** 2, dim=0)
                actionss = torch.repeat_interleave(actions, repeats=sample_times ** 2, dim=0)
                samples = denormalize_observation_delta(samples, self.statistics)
                samples += observationss
                if self.sas:
                    if self.prob_rwd:
                        rewards, rwd_var = self.reward_network(observationss, actionss, samples)
                        epis_uncert = torch.var(rewards, dim=0).item()
                        rwd_var = rwd_var.squeeze().detach().cpu().numpy()
                        alea_uncert = (rwd_var ** 2).mean(axis=0) ** 0.5
                        epis_uncert = np.minimum(epis_uncert, 10e3)
                        alea_uncert = np.minimum(alea_uncert, 10e3)
                        uncert_rwd = ((epis_uncert ** 2) + (alea_uncert ** 2)) ** 0.5
                    else:
                        rewards = self.reward_network(observationss, actionss, samples)
                        uncert_rwd = torch.var(rewards, dim=0).item()
                else:
                    if self.prob_rwd:
                        rewards, rwd_var = self.reward_network(samples, actionss)
                        epis_uncert = torch.var(rewards, dim=0).item()
                        rwd_var = rwd_var.squeeze().detach().cpu().numpy()
                        alea_uncert = (rwd_var ** 2).mean(axis=0) ** 0.5
                        epis_uncert = np.minimum(epis_uncert, 10e3)
                        alea_uncert = np.minimum(alea_uncert, 10e3)
                        uncert_rwd = ((epis_uncert ** 2) + (alea_uncert ** 2)) ** 0.5
                    else:
                        rewards = self.reward_network(samples, actionss)
                        uncert_rwd = torch.var(rewards, dim=0).item()
        return uncert, uncert_rwd

    def train_together(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, ):
        normalized_states = normalize_observation(states, self.statistics)
        s_n_a = torch.cat((normalized_states, actions), dim=1)
        # pred = self.bnn(s_n_a, pred_type="glm", link_approx="probit")
        pred = self.world_model(s_n_a)
        mean, var = pred[:, :self.observation_size], pred[:, self.observation_size:]
        # # Sampling does not working well.
        sample_times = 100
        dist = torch.distributions.Normal(mean, var)
        samples = dist.sample([sample_times]).squeeze()
        states = torch.repeat_interleave(states.unsqueeze(dim=0), sample_times, dim=0)
        actions = torch.repeat_interleave(actions.unsqueeze(dim=0), sample_times, dim=0)
        rewards = torch.repeat_interleave(rewards.unsqueeze(dim=0), sample_times, dim=0)
        actions = torch.reshape(actions, (actions.shape[0] * actions.shape[1], actions.shape[2]))
        states = torch.reshape(states, (states.shape[0] * states.shape[1], states.shape[2]))
        samples = torch.reshape(samples, (samples.shape[0] * samples.shape[1], samples.shape[2]))
        rewards = torch.reshape(rewards, (rewards.shape[0] * rewards.shape[1], rewards.shape[2]))

        samples = denormalize_observation_delta(samples, self.statistics)
        samples += states
        samples = samples.detach()

        if self.prob_rwd:
            if self.sas:
                rwd_mean, rwd_var = self.reward_network(states, actions, samples)
            else:
                rwd_mean, rwd_var = self.reward_network(samples, actions)
            rwd_loss = F.gaussian_nll_loss(rwd_mean, rewards, rwd_var)
        else:
            if self.sas:
                rwd_mean = self.reward_network(states, actions, samples)
            else:
                rwd_mean = self.reward_network(samples, actions)
            rwd_loss = F.mse_loss(rwd_mean, rewards)
        self.reward_optimizer.zero_grad()
        rwd_loss.backward()
        self.reward_optimizer.step()

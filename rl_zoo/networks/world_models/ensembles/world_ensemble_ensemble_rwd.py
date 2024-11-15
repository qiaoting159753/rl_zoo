import math
import torch
import torch.utils
from rl_zoo.networks.world_models.deterministic import Single_PNN
from rl_zoo.utils.helpers import normalize_observation, denormalize_observation_delta
import numpy as np


class Ensemble_Dyna_Ensemble_Reward:
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 device: str,
                 num_models: int = 5,
                 l_r: float = 0.001,
                 boost_inter: int = 3,
                 sas: bool = True,
                 prob_rwd: bool = True,
                 hidden_size=None):
        if hidden_size is None:
            hidden_size = [128, 128, 128]
        self.num_models = num_models
        self.boost_inter = boost_inter
        self.update_counter = 0
        self.world_models = [Single_PNN(observation_size=observation_size,
                                        num_actions=num_actions,
                                        l_r=l_r,
                                        device=device,
                                        hidden_size=hidden_size,
                                        sas=sas,
                                        prob_rwd=prob_rwd) for _ in range(num_models)]

    def set_statistics(self, statistics: dict) -> None:
        for world_model in self.world_models:
            world_model.set_statistics(statistics)

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        preds = []
        for world_model in self.world_models:
            a, b, _, _ = world_model.pred_next_states(observation, actions)
            preds.append(a)
        preds = torch.vstack(preds).squeeze()
        return torch.mean(preds, dim=0, keepdim=True), None, None, None

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        index = int(math.floor(self.update_counter / self.boost_inter))
        self.world_models[index].train_world(states, actions, next_states)
        self.update_counter += 1
        self.update_counter %= self.boost_inter * self.num_models

    def pred_rewards(self, observation: torch.Tensor,
                     action: torch.Tensor, next_observation: torch.Tensor):
        """
        predict reward based on current observation and action and next state
        """
        preds = []
        for world_model in self.world_models:
            a, _ = world_model.pred_rewards(observation, action, next_observation)
            preds.append(a)
        preds = torch.vstack(preds)
        preds = torch.mean(preds, dim=0, keepdim=True)
        return preds, None

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        index = int(math.floor(self.update_counter / self.boost_inter))
        self.world_models[index].train_reward(states, actions, next_states, rewards)


    def train_together(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        index = int(math.floor(self.update_counter / self.boost_inter))
        self.world_models[index].train_together(states=states, actions=actions, rewards=rewards)


    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        means = []
        vars_s = []
        pred_rwd_means = []
        pred_rwd_vars = []
        sample_times = 20
        for model in self.world_models:
            normalized_state = normalize_observation(observation, model.statistics)
            mean, var = model.world_model.forward(normalized_state, actions)
            dist = torch.distributions.Normal(mean, var)
            sampled_delta = dist.sample([sample_times])
            sampled_delta = sampled_delta.squeeze()
            denorm_delta = denormalize_observation_delta(sampled_delta, model.statistics)
            pred_next = denorm_delta + observation
            rwd_mean, rwd_var = model.pred_rewards(torch.repeat_interleave(observation, dim=0, repeats=sample_times),
                                                   torch.repeat_interleave(actions, dim=0, repeats=sample_times),
                                                   pred_next)
            pred_rwd_means.append(rwd_mean)
            pred_rwd_vars.append(rwd_var)
            means.append(mean)
            vars_s.append(var)

        noises = torch.stack(vars_s).cpu().squeeze().detach().numpy()
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        all_means = torch.stack(means).cpu().squeeze().detach().numpy()
        epistemic = all_means.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc)

        rwd_noises = torch.vstack(pred_rwd_vars).cpu().detach().numpy()
        rwd_aleatoric = (rwd_noises.squeeze() ** 2).mean(axis=0) ** 0.5
        rwd_all_means = torch.vstack(pred_rwd_means).cpu().detach().numpy()
        rwd_epistemic = rwd_all_means.squeeze().var(axis=0) ** 0.5
        rwd_aleatoric = np.minimum(rwd_aleatoric, 10e3)
        rwd_epistemic = np.minimum(rwd_epistemic, 10e3)
        rwd_total_unc = (rwd_aleatoric ** 2 + rwd_epistemic ** 2) ** 0.5
        rwd_uncert = np.mean(rwd_total_unc)
        return uncert, rwd_uncert


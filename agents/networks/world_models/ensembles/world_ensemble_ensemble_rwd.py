import math
import torch
import torch.utils
from agents.networks.world_models.deterministic import Single_PNN
from utils.helpers import normalize_observation_delta, normalize_observation, denormalize_observation_delta
import numpy as np


class Ensemble_Dyna_Ensemble_Reward:
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 num_models: int,
                 l_r: float,
                 device: str,
                 boost_inter: int,
                 sas: bool = True,
                 prob_rwd: bool = False,
                 hidden_size: int = 128, ):
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        means = []
        vars_s = []
        for model in self.world_models:
            normalized_state = normalize_observation(observation, model.statistics)
            mean, var = model.world_model.forward(normalized_state, actions)
            means.append(mean)
            vars_s.append(var)
        noises = torch.stack(vars_s).squeeze().detach().numpy()
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        all_means = torch.stack(means).squeeze().detach().numpy()
        epistemic = all_means.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc)

        return uncert, 0.0
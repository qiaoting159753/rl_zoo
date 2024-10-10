import torch
import torch.nn.functional as F
import torch.utils
from torch import optim
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, normalize_observation, denormalize_observation_delta
from agents.networks.world_models.simple import Probabilistic_Dynamics


class Single_PNN(World_Model):
    """
    This class consist of an ensemble of all components for critic update.
    Q_label = REWARD + gamma * (1 - DONES) * Q(NEXT_STATES).
    """
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 l_r: float,
                 device: str,
                 hidden_size: int = 256,
                 sas: bool = True,
                 prob_rwd: bool = False):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        self.prob_rwd = prob_rwd
        self.sas = sas
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.device = device
        self.world_model = Probabilistic_Dynamics(observation_size=observation_size, num_actions=num_actions,
                                                  hidden_size=hidden_size)
        self.world_optimizers = optim.Adam(self.world_model.parameters(), lr=l_r)
        self.world_model.to(self.device)

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        """
        Predict the next state based on the current state and action.
        """
        # Predict delta
        normalized_observation = normalize_observation(observation, self.statistics)
        n_mean, n_var = self.world_model.forward(normalized_observation, actions)
        prediction = denormalize_observation_delta(n_mean, self.statistics)
        prediction += observation
        return prediction, None, n_mean, n_var

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train the world with S, A, SN. Different sub-batch.
        """
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_state = normalize_observation(states, self.statistics)
        n_mean, n_var = self.world_model.forward(normalized_state, actions)
        model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
        self.world_optimizers.zero_grad()
        model_loss.backward()
        self.world_optimizers.step()

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        """
        Estimate uncertainty.
        """
        normalized_state = normalize_observation(observation, self.statistics)
        mean, var = self.world_model.forward(normalized_state, actions)
        uncert = torch.mean(var.squeeze()).item()

        # Reward Uncertainty
        sample_times = 100
        dist = torch.distributions.Normal(mean, var)
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
                aleatoric = (rwd_var.cpu().detach().numpy() ** 2).mean(axis=0) ** 0.5
                uncert_rwd = (epis_uncert + aleatoric) ** 0.5
            else:
                rewards = self.reward_network(observationss, actionss, samples)
                uncert_rwd = torch.var(rewards, dim=0).item()
        else:
            if self.prob_rwd:
                rewards, rwd_var = self.reward_network(samples, actionss)
                epis_uncert = torch.var(rewards, dim=0).item()
                aleatoric = (rwd_var.cpu().detach().numpy() ** 2).mean(axis=0) ** 0.5
                uncert_rwd = epis_uncert + aleatoric
            else:
                rewards = self.reward_network(samples, actionss)
                uncert_rwd = torch.var(rewards, dim=0).item()
        return uncert, uncert_rwd

    def train_together(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, ):
        normalized_state = normalize_observation(states, self.statistics)
        mean, var = self.world_model.forward(normalized_state, actions)
        sample_times = 100
        dist = torch.distributions.Normal(mean, var)
        samples = (dist.sample([sample_times]))
        samples = samples.squeeze()

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

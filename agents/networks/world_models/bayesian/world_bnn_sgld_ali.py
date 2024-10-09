from __future__ import division
import torch
import copy
from agents.networks.world_models import World_Model
from utils.helpers import denormalize_observation_delta, normalize_observation, normalize_observation_delta
from agents.networks.world_models.bayesian.bayesian_sgld_sample_ali import CustomizedMLP, SGLD_Sampler
from agents.networks.world_models.simple import Probabilistic_Dynamics


class Bayesian_World_Model_SGLD_JA(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r,
                 hidden_size,
                 device,
                 sas,
                 prob_rwd):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model = CustomizedMLP(observation_size+num_actions, [128, 128], 2 * observation_size)
        self.world_model.to(self.device)
        self.sampler = SGLD_Sampler(self.world_model)

        # self.world_model = Probabilistic_Dynamics(observation_size=observation_size, num_actions=num_actions,
        #                                           hidden_size=hidden_size)
        # self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r)
        # self.world_sampler = SGLD(self.world_model.parameters(), lr=0.00001)

        self.weight_set_samples = []
        self.counter = 0
        self.trajectory_length = 10

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Predict delta
        normalized_observation = normalize_observation(observation, self.statistics)
        query_input = torch.cat((normalized_observation, actions), dim=1)
        n_mean, n_var = self.world_model.forward(query_input)
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
        Train for On-policy flush training use.
        :param states:
        :param actions:
        :param next_states:
        """
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_state = normalize_observation(states, self.statistics)
        iiiinput = torch.cat((normalized_state, actions), dim=1)
        self.world_model.current_input = iiiinput
        self.world_model.current_output = delta_targets_normalized
        self.sampler.sample_chains()

        # target = next_states - states
        # delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        # normalized_state = normalize_observation(states, self.statistics)
        # n_mean, n_var = self.world_model.forward(normalized_state, actions)
        # self.world_optimizers.zero_grad()
        # model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
        # model_loss.backward()
        # self.world_optimizers.step()
        # for _ in range(self.trajectory_length):
        #     pred_mean, pred_var = self.world_model(normalized_state, actions)
        #     model_loss = F.gaussian_nll_loss(input=pred_mean, target=delta_targets_normalized, var=pred_var).mean()
        #     self.world_sampler.zero_grad()
        #     model_loss.backward()
        #     self.world_sampler.step()
        #     self.save_sampled_net(100)









    # def save_sampled_net(self, max_samples=100):
    #     """
    #     Sample the network parameters with optimizer?
    #     :param max_samples:
    #     :return:
    #     """
    #     if len(self.weight_set_samples) >= max_samples:
    #         self.weight_set_samples.pop(0)
    #     self.weight_set_samples.append(copy.deepcopy(self.world_model.state_dict()))
    #     return None

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        # normalized_state = normalize_observation(observation, self.statistics)
        # length = len(self.weight_set_samples)
        # uncert = 0.0
        # if length > 0:
        #     Nsamples = len(self.weight_set_samples)
        #     sample_times = 10
        #     predictions = []
        #     # iterate over all saved weight configuration samples
        #     for idx, weight_dict in enumerate(self.weight_set_samples):
        #         if idx == Nsamples:
        #             break
        #         self.world_model.load_state_dict(weight_dict)
        #         n_mean_delta, normalized_var = self.world_model(normalized_state, actions)
        #         sample1 = torch.distributions.Normal(n_mean_delta, normalized_var).sample([sample_times])
        #         sample1 = sample1.squeeze()
        #         predictions.append(sample1)
        #     predictions = torch.stack(predictions)
        #     predictions = torch.reshape(predictions, (length * sample_times, self.observation_size))
        #     uncert = torch.mean(torch.var(predictions, dim=0)).item()
        return 0.0, 0.0
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from rl_zoo.networks.world_models import World_Model
from rl_zoo.utils import ConditionalDiagGaussian
from rl_zoo.utils import MaskedAffineAutoregressive, normalize_observation, normalize_observation_delta, \
    denormalize_observation_delta, Permute


class Conditional_NVP_World_Model(World_Model):
    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            l_r: float,
            device: str,
            hidden_size: int = 128,
    ):
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.l_r = l_r
        self.device = device
        self.hidden_size = hidden_size
        self.statistic = {}
        self.world_model = Conditional_NVP_Flows(self.observation_size, self.num_actions)
        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)
        self.reward_optimizers = optim.Adam(self.reward_model.parameters(), lr=l_r)

    def set_statistics(self, statistics: dict) -> None:
        """
        Update all statistics for normalization for all world models and the
        ensemble itself.

        :param (Dictionary) statistics:
        """
        for i in statistics:
            statistics[i] = torch.FloatTensor(statistics[i])
        self.statistic = statistics
        self.world_model.statistics = statistics

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
        self.world_model.train_net(states, actions, next_states)

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_next, _, _ = self.world_model.forward(observation, actions)
        return pred_next, torch.zeros(observation.shape), torch.zeros(observation.shape), torch.zeros(observation.shape)

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        """
        Estimate next state uncertainty and reward uncertainty.

        :param observation:
        :param actions:
        :return:
        """
        # logging.info("Not Implemented")
        pred_next, pred_z, _ = self.world_model.forward(observation, actions)
        normalized_obs = normalize_observation(observation, self.statistic)
        target_z_ = torch.cat((normalized_obs, actions), dim=1)
        z_,_ = self.world_model.reverse(pred_z)
        mse_loss = F.mse_loss(z_, target_z_).item()
        # mse_loss = np.sum(abs(z_.detach().numpy() - target_z_.detach().numpy()))
        return mse_loss, 0.0

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
        # reward_loss = F.mse_loss(rwd_mean, sub_rewards)
        reward_loss = F.gaussian_nll_loss(input=rwd_mean, target=rewards, var=rwd_var).mean()
        reward_loss.backward()
        self.reward_optimizers.step()

    def pred_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return:
        """
        pred_reward, reward_var = self.reward_model.forward(observation, action, next_observation)
        return pred_reward, None, reward_var


class Conditional_NVP_Flows:
    """
    Conditional NVP.
    """
    # Forward KLD: inverse back, -log_q - init_log.
    # Reverse KLD: forward, + init_log - log_det, loss = (mean - beta * mean).
    # total_params = sum(p.numel() for p in self.flows.parameters())
    # print("Normalizing Flows Model One model No. Parameters: ")
    # print(total_params)
    def __init__(self, state_dim, act_dim):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.shape = (state_dim + act_dim,)
        # Define a nf
        num_layers = 32
        flows = []
        self.q_0 = ConditionalDiagGaussian(self.shape)
        for i in range(num_layers):
            mask = MaskedAffineAutoregressive(state_dim + act_dim, hidden_features=128, num_blocks=1)
            # Swap dimensions
            flows.append(mask)
            flows.append(Permute(state_dim + act_dim, mode='swap'))
        self.flows = nn.ModuleList(flows)
        self.optimizer = optim.Adam(self.flows.parameters(), lr=0.00001)
        self.statistics = dict()

    def forward(self, states, actions):
        """

        :param states:
        :param actions:
        :return:
        """
        normalized_obs = normalize_observation(states, self.statistics)
        s_a = torch.cat((normalized_obs, actions), dim=1)
        log_dets = 0
        z_, log_det = self.q_0.forward(s_a, log_scale=torch.ones(s_a.shape) * 0.1)
        log_dets += log_det
        for flow in self.flows:
            z_, log_det = flow.forward(z_)
            log_dets += log_det
        pred = z_[:, 0:self.state_dim]
        pred_delta = denormalize_observation_delta(pred, self.statistics)
        pred_next = pred_delta + states
        return pred_next, z_, log_dets

    def train_net(self, states, actions, next_states):
        """
        :param states:
        :param actions:
        :param next_states:
        """
        # Target is the normalized diff
        target = (next_states - states)
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        s_n_a = torch.cat((delta_targets_normalized, actions), dim=1)
        _, z_end, log_dets = self.forward(states, actions)
        # Reverse KLD: Log_q - Log_p = (Log_q0 - forward_log_det) - (-MSE)
        mse_loss = F.mse_loss(z_end, s_n_a, reduction="sum")
        loss = torch.mean(-1 * log_dets + mse_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reverse(self, z_):
        """
        Reverse to the start
        :param z_:
        :return:
        """
        # Reverse
        reverse_log_dets = 0
        for i in range(len(self.flows) - 1, -1, -1):
            z_, reverse_log_det = self.flows[i].inverse(z_)
            reverse_log_dets += reverse_log_det
        return z_, reverse_log_dets

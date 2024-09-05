import torch
from torch import nn

from agents.utils import MLP


class TQC_Critic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_quantiles: int,
        num_critics: int,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256, 256, 256]

        self.q_networks = []
        self.num_quantiles = num_quantiles
        self.num_critics = num_critics

        for i in range(self.num_critics):
            critic_net = MLP(
                observation_size + num_actions, hidden_size, self.num_quantiles
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.q_networks.append(critic_net)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        network_input = torch.cat((state, action), dim=1)
        quantiles = torch.stack(
            tuple(critic(network_input) for critic in self.q_networks), dim=1
        )
        return quantiles
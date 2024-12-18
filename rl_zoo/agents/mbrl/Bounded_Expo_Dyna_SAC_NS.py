import logging
import numpy as np
import torch
import torch.nn.functional as F
from rl_zoo.agents.mbrl import Dyna_SAC_NS
from rl_zoo.networks.world_models import (
    World_Model,
)


class Bounded_Expo_Dyna_SAC_NS(Dyna_SAC_NS):
    """
    Dyna of SAC with Next state to predict rewards.
    """
    def __init__(self,
                 actor_network: torch.nn.Module,
                 critic_network: torch.nn.Module,
                 world_network: World_Model,
                 gamma: float,
                 tau: float,
                 action_num: int,
                 actor_lr: float,
                 critic_lr: float,
                 alpha_lr: float,
                 num_samples: int,
                 horizon: int,
                 exploration_samples: int,
                 alpha: float,
                 device: torch.device,
                 train_reward: bool,
                 train_both: bool,
                 gripper: bool):
        super().__init__(actor_network=actor_network,
                         critic_network=critic_network,
                         world_network=world_network,
                         gamma=gamma,
                         tau=tau,
                         action_num=action_num,
                         actor_lr=actor_lr,
                         critic_lr=critic_lr,
                         alpha_lr=alpha_lr,
                         num_samples=num_samples,
                         horizon=horizon,
                         device=device,
                         train_reward=train_reward,
                         train_both=train_both,
                         gripper=gripper)
        logging.info("---------------------------------------------------------------")
        logging.info("----I am runing the Bounded_Exploration_Dyna_SAC_NS Agent! ----")
        logging.info("---------------------------------------------------------------")
        self.exploration_samples = exploration_samples
        self.alpha = alpha
        self.set_stat = False

    def select_action_from_policy(
            self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluation is False:
                if self.alpha == 0:
                    (action, _, _) = self.actor_net(state_tensor)
                else:
                    if self.set_stat:
                        multi_state_tensor = torch.repeat_interleave(state_tensor, self.exploration_samples, dim=0)
                        (multi_action, multi_log_pi, _) = self.actor_net(multi_state_tensor)
                        # Estimate uncertainty
                        # [6, 10, 17]
                        _, _, nstate_means, nstate_vars = self.world_model.pred_next_states(observation=multi_state_tensor, actions=multi_action)
                        # [10, 17]
                        aleatoric = torch.mean(nstate_vars ** 2, dim=0) ** 0.5
                        epistemic = torch.var(nstate_means, dim=0) ** 0.5
                        aleatoric = torch.clamp(aleatoric, max=10e3)
                        epistemic = torch.clamp(epistemic, max=10e3)
                        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
                        uncert = torch.mean(total_unc, dim=1)
                        multi_log_pi = multi_log_pi.squeeze()
                        policy_dist = F.softmax(multi_log_pi, dim=0)
                        world_dist = F.softmax(uncert, dim=0)
                        final_dist = (1 - self.alpha) * policy_dist + self.alpha * world_dist
                        final_dist = F.softmax(final_dist, dim=0)
                        candi = torch.argmax(final_dist)
                        # new_dist = torch.distributions.Categorical(final_dist)
                        # candi = new_dist.sample([5]).squeeze()
                        action = multi_action[candi]
                    else:
                        (action, _, _) = self.actor_net(state_tensor)
            else:
                (_, _, action) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def set_statistics(self, stats: dict) -> None:
        self.world_model.set_statistics(stats)
        self.set_stat = True

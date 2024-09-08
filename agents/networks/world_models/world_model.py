import logging
import torch


class World_Model:
    """
    World Model

    """

    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            num_models: int,
            lr: float,
            device: str,
            hidden_size: int = 128,
    ):
        logging.info("Not Implemented")

    def set_statistics(self, statistics: dict) -> None:
        """
        Update all statistics for normalization for all world models and the
        ensemble itself.

        :param (Dictionary) statistics:
        """
        logging.info("Not Implemented")

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:
        """
        Estimate next state uncertainty and reward uncertainty.

        :param observation:
        :param actions:
        :return:
        """
        logging.info("Not Implemented")
        return 0.0, 0.0

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
        logging.info("Not Implemented")

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
        logging.info("Not Implemented")

    def pred_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return:
        """
        logging.info("Not Implemented")
        return torch.zeros((1,)), torch.zeros((1,))

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logging.info("Not Implemented")
        return torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))

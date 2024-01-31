import numpy as np
import torch


class ReplayBuffer:
    """Buffer to store environment transitions."""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.statistics = None
        self.obs_shape = state_dim
        self.action_shape = action_dim
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.full = False
        self.obses = np.empty((capacity, *state_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, *state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        """
        Add new transition in the buffer.
        :param obs:
        :param action:
        :param reward:
        :param next_obs:
        :param done:
        """
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """
        Randomly Sample transitions from stored data
        :param batch_size:
        :return:
        """
        idxs = np.random.randint(0, (self.capacity-1) if self.full else (self.idx-1),
                                 size=batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_actions = torch.as_tensor(self.actions[idxs+1], device=self.device)
        next_rewards = torch.as_tensor(self.rewards[idxs+1], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones, next_actions, next_rewards

    def get_statistics(self):
        """
        Compute the statisitics for world model normalization.
        :return:
        """
        obs_torch = torch.from_numpy(self.obses).float().to(self.device)
        deltas_torch = torch.from_numpy(self.next_obses - self.obses).float().to(self.device)
        statistics = {
            'ob_mean': obs_torch.mean(dim=0) + 0.0001,
            'ob_std': obs_torch.std(dim=0) + 0.0001,
            'delta_mean': deltas_torch.mean(dim=0) + 0.0001,
            'delta_std': deltas_torch.std(dim=0)+0.0001
        }
        return statistics
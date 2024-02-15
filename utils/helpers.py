import torch
import torch.nn as nn
import random
import numpy as np
import os


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def normalize_obs(obs, statistics):
    """
    Normalize the obs for model-based training.

    :param obs:
    :param statistics:
    :return:
    """
    return (obs - statistics["ob_mean"]) / statistics["ob_std"]


def unnormalize_obs_deltas(normalized_deltas, statistics):
    """
    Unnormalize the next - current observations to resume

    :param normalized_deltas:
    :param statistics:
    :return:
    """
    return (normalized_deltas * statistics["delta_std"]) + statistics["delta_mean"]


def normalize_obs_deltas(deltas, statistics):
    """

    :param deltas:
    :param statistics:
    :return:
    """
    return (deltas - statistics["delta_mean"]) / statistics["delta_std"]


def set_seed(seed):
    """
    Set a new seed to all environment, neuarl network initializations.

    :param seed:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def soft_update(local_model, target_model, tau):
    """
    Update one network toward another a little.

    :param local_model:
    :param target_model:
    :param tau:
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
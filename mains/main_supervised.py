import numpy as np
import torch
import torch.nn.functional as F
from rl_zoo.networks.world_models.ensembles import Ensemble_Dyna_One_Reward
from rl_zoo.networks.mfrl.common import Actor
from rl_zoo.networks.mfrl.sac import SAC_Critic
from rl_zoo.agents.mbrl import Dyna_SAC_NS

states = np.load("/Users/tonyq/Downloads/saved_states.npy")
actions = np.load("/Users/tonyq/Downloads/saved_actions.npy")
next_states = np.load("/Users/tonyq/Downloads/saved_next_states.npy")

diff_states = next_states - states
observation_mean = np.mean(states, axis=0) + 0.00001
observation_std = np.std(states, axis=0) + 0.00001
delta_mean = np.mean(diff_states, axis=0) + 0.00001
delta_std = np.std(diff_states, axis=0) + 0.00001

statistics = {
    "observation_mean": observation_mean,
    "observation_std": observation_std,
    "delta_mean": delta_mean,
    "delta_std": delta_std,
}

states = torch.FloatTensor(states)
actions = torch.FloatTensor(actions)
next_states = torch.FloatTensor(next_states)

batch_size = states.shape[0]
half_batch = int(batch_size / 2)

train_states = states[:half_batch, :]
train_actions = actions[:half_batch, :]
train_next_states = next_states[:half_batch, :]

test_states = states[half_batch:, :]
test_actions = actions[half_batch:, :]
test_next_states = next_states[half_batch:, :]

model = Ensemble_Dyna_One_Reward(observation_size=16,
                                 num_actions=4,
                                 device='cpu')

model.set_statistics(statistics)

actor = Actor(observation_size=16, num_actions=4)
critic = SAC_Critic(observation_size=16, num_actions=4)

agent = Dyna_SAC_NS(actor_network=actor,
                    critic_network=critic,
                    world_network=model,
                    num_samples=10,
                    action_num=4,
                    alpha_lr=3e-4,
                    horizon=1,
                    gamma=0.99,
                    tau=0.005,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    device='cpu')

for i in range(10000):
    model.train_world(train_states, train_actions, train_next_states)
    # agent.world_model.train_world(train_states, train_actions, train_next_states)

    pred_next_states, _, _, _, = model.pred_next_states(test_states, test_actions)
    mse_loss = torch.sqrt(torch.sum((pred_next_states - test_next_states) ** 2)) / (
                test_states.shape[0] * test_states.shape[1])
    print("--------------------")
    print(i)
    print(mse_loss.item())

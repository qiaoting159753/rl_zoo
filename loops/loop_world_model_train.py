from envs import DMCSEnvironment
from networks.mbrl import Ensemble_World_Reward_GAN
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

env = DMCSEnvironment("cheetah", "run")

world_model = Ensemble_World_Reward_GAN(state_dim=env.observation_space,
                                        action_dim=env.action_num,
                                        num_models=5)

epochs = 500
batch_size = 128


states = []
actions = []
next_states = []
rewards = []
mse_loss = []

for i in tqdm(range(epochs)):
    state = env.reset()
    for j in range(1000):
        action = np.random.uniform(env.min_action_value, env.max_action_value,
                                   (env.action_num,))
        next_state, reward, done, info = env.step(action)
        print(reward)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)

        if len(states) == batch_size:
            # Set statistics.
            states_tensor = torch.FloatTensor(np.array(states[:len(states)-1]))
            actions_tensor = torch.FloatTensor(np.array(actions[:len(states)-1]))
            rewards_tensor = torch.FloatTensor(np.array(rewards[:len(states)-1])).unsqueeze(dim=1)
            next_states_tensor = torch.FloatTensor(np.array(next_states[:len(states)-1]))
            next_actions_tensor = torch.FloatTensor(np.array(actions[1:len(states)]))
            next_rewards_tensor = torch.FloatTensor(np.array(rewards[1:len(states)])).unsqueeze(dim=1)

            # Set statistics before all happens.
            deltas_torch = next_states_tensor - states_tensor
            statistics = {
                'ob_mean': states_tensor.mean(dim=0) + 0.000001,
                'ob_std': states_tensor.std(dim=0) + 0.000001,
                'delta_mean': deltas_torch.mean(dim=0) + 0.000001,
                'delta_std': deltas_torch.std(dim=0) + 0.000001
            }
            world_model.set_statistics(statistics)

            # Evaluate before
            pred, _, _, _ = world_model.pred_next_states(states_tensor,
                                                         actions_tensor)
            mse_loss.append(torch.mean(F.mse_loss(pred, next_states_tensor)).item())

            # Then training.
            world_model.train_world(states_tensor, actions_tensor,
                                    rewards_tensor, next_states_tensor,
                                    next_actions_tensor, next_rewards_tensor)

            states =[]
            actions = []
            next_states = []
            rewards = []

# eval_array = np.array(mse_loss)
# np.savetxt("GAN_MSE.csv", eval_array, delimiter=",")

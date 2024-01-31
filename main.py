import torch
import gymnasium as gym
from memories import ReplayBuffer
from agents.mbrl.mbrl_sac import MBRL_SAC
from networks.soft_actor import DiagGaussianActor
from networks.double_critic import DoubleQCritic
from networks.mbrl.ensemble_world import Ensemble_World_Reward
from train_loops.runner import Runner


def main():
    """
    Create all parts and get it to run
    """
    env_name = "HalfCheetah-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    num_models = 5
    capacity = 1000000

    actor = DiagGaussianActor(state_dim, action_dim, hidden_dim)
    critic = DoubleQCritic(state_dim, action_dim, hidden_dim)
    world_model = Ensemble_World_Reward(state_dim, action_dim, num_models)
    memory = ReplayBuffer(env.observation_space.shape, env.action_space.shape,
                          capacity, device)

    agent = MBRL_SAC(actor, critic, world_model)

    runner = Runner(env, agent, memory)


if __name__ == "__main__":
    main()

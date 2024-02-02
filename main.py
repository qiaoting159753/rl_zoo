import torch
import gymnasium as gym
from memories import ReplayBuffer
from agents.mbrl.mbrl_sac import MBRL_SAC
from networks.soft_actor import Actor
from networks.double_critic import DoubleQCritic
from networks.mbrl.ensemble_world import Ensemble_World_Reward
from train_loops.trainer import Trainer


def main():
    """
    Create all parts and get it to run
    """
    use_dyna = True
    use_critic_steve = False
    use_critic_mve = False
    use_actor_mve = False
    use_actor_pg = False
    use_bound = False

    env_name = "HalfCheetah-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    num_models = 5
    capacity = 1000000

    actor = Actor(state_dim, action_dim)
    critic = DoubleQCritic(state_dim, action_dim)
    world_model = Ensemble_World_Reward(state_dim, action_dim, num_models)

    memory = ReplayBuffer(env.observation_space.shape, env.action_space.shape,
                          capacity, device)

    agent = MBRL_SAC(actor, critic, world_model,
                     state_dim=state_dim,
                     action_dim=action_dim,
                     actor_lr=3e-4,
                     critic_lr=3e-4,
                     alpha_lr=3e-4,
                     gamma=0.99,
                     tau=0.005,
                     horizon=3,
                     use_dyna=use_dyna,
                     use_critic_steve=use_critic_steve,
                     use_critic_mve=use_critic_mve,
                     use_actor_mve=use_actor_mve,
                     use_actor_pg=use_actor_pg,
                     use_bound=use_bound,
                     device=device)

    runner = Trainer(env, agent, memory,
                     use_dyna=use_dyna,
                     use_critic_steve=use_critic_steve,
                     use_critic_mve=use_critic_mve,
                     use_actor_mve=use_actor_mve,
                     use_actor_pg=use_actor_pg,
                     use_bound=use_bound)

    runner.train_loop()


if __name__ == "__main__":
    main()

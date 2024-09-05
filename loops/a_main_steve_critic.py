import torch
import logging
from envs import DMCSEnvironment
from memories import MemoryFactory
from agents.mbrl import STEVE_MEAN
from networks.mfrl.actor import Actor
from networks.mfrl.critic import Critic
from networks.mbrl import EnsembleWorldRewardDone
from loops.a_trainer import Trainer
from utils import set_seed


def main():
    """
    Create all parts and get it to run
    """
    # Training settings.
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    seed = 15
    set_seed(seed)

    # Environment settings.
    domain_name = "cheetah"
    task_name = "run"

    env = DMCSEnvironment(domain_name, task_name)
    action_dim = env.action_num
    state_dim = env.observation_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Algorithm settings.
    alg = "steve_mean"
    name = alg

    num_models = 5
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)

    world_model = EnsembleWorldRewardDone(observation_size=state_dim,
                                          num_actions=action_dim,
                                          num_world_models=num_models,
                                          num_reward_models=num_models,
                                          lr=0.001,
                                          device="cpu")

    memory = MemoryFactory().create_memory()

    agent = STEVE_MEAN(actor, critic, world_model,
                       device=device,
                       action_num=action_dim,
                       actor_lr=3e-4,
                       critic_lr=3e-4,
                       alpha_lr=3e-4,
                       gamma=0.99,
                       tau=0.005,
                       horizon=2,
                       L=4)

    runner = Trainer(env, agent, memory, device=device, name=name, logger=log)
    runner.train_loop()


if __name__ == "__main__":
    main()

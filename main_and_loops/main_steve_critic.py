import torch
import logging
from envs import DMCSEnvironment
from memories import MemoryBuffer
from agents.mbrl import MBRL_STEVE_CRITIC
from networks.actor import Actor
from networks.critic import Critic
from networks.mbrl import Ensemble_World_Reward
from main_and_loops.trainer import Trainer
from utils import set_seed


def main():
    """
    Create all parts and get it to run
    """
    # Training settings.
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    seed = 10
    set_seed(seed)
    generate_results = True

    # Environment settings.
    domain_name = "cheetah"
    task_name = "run"
    env = DMCSEnvironment(domain_name, task_name)
    action_dim = env.action_num
    state_dim = env.observation_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Algorithm settings.
    alg = "sac"
    name = alg

    num_models = 5
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    world_model = Ensemble_World_Reward(state_dim, action_dim, num_models)
    memory = MemoryBuffer()

    agent = MBRL_STEVE_CRITIC(actor, critic, world_model, device=device,
                              action_num=action_dim,
                              actor_lr=3e-4,
                              critic_lr=3e-4,
                              alpha_lr=3e-4,
                              gamma=0.99,
                              tau=0.005,
                              horizon=2,
                              num_samples=3, )

    runner = Trainer(generate_results, env, agent, memory, name=name,
                     device=device, use_mbrl=True, logger=log)

    runner.train_loop()


if __name__ == "__main__":
    main()

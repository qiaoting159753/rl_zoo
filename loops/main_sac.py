import torch
import logging
from envs import DMCSEnvironment
from memories import ReplayBuffer
from agents import SAC
from networks.mfrl.actor import Actor
from networks.mfrl.critic import DoubleQCritic
from loops.a_trainer import Trainer
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
    use_mbrl = False
    alg = "sac"
    name = alg

    capacity = 1000000

    actor = Actor(state_dim, action_dim)
    critic = DoubleQCritic(state_dim, action_dim)
    memory = ReplayBuffer((state_dim,), (action_dim,),
                          capacity, device)

    agent = SAC(actor, critic, device=device,
                state_dim=state_dim,
                action_dim=action_dim,
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                gamma=0.99,
                tau=0.005)

    runner = Trainer(generate_results, env, agent, memory, name=name,
                     device=device, use_mbrl=use_mbrl, logger=log)

    runner.train_loop()


if __name__ == "__main__":
    main()

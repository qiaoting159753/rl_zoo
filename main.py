import torch
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setStream(tqdm)
log.addHandler(handler)

from envs import DMCSEnvironment
from memories import ReplayBuffer
from agents.mbrl import MBRL_DYNA_SAC, MBRL_STEVE_SAC
from networks.soft_actor import Actor
from networks.double_critic import DoubleQCritic
# from networks.distribution_Q import DoubleDistributionalQCritic
from networks.mbrl.ensemble_world import Ensemble_World_Reward
from train_loops.trainer import Trainer
from utils import set_seed


def main():
    """
    Create all parts and get it to run
    """
    # Training settings.
    seed = 10
    set_seed(seed)
    generate_results = False
    use_bound = False

    # Environment settings.
    domain_name = "cheetah"
    task_name = "run"
    env = DMCSEnvironment(domain_name, task_name)
    action_dim = env.action_num
    state_dim = env.observation_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Algorithm settings.
    use_dyna = True
    name = "dyna_norm"
    num_models = 5
    capacity = 1000000

    actor = Actor(state_dim, action_dim)
    critic = DoubleQCritic(state_dim, action_dim)
    # critic = DoubleDistributionalQCritic(state_dim, action_dim)
    world_model = Ensemble_World_Reward(state_dim, action_dim, num_models)
    memory = ReplayBuffer((state_dim,), (action_dim,),
                          capacity, device)

    agent = MBRL_DYNA_SAC(actor, critic, world_model, device=device,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          actor_lr=3e-4,
                          critic_lr=3e-4,
                          alpha_lr=3e-4,
                          gamma=0.99,
                          tau=0.005,
                          horizon=3,
                          sample_times=128,
                          use_bound=use_bound)

    runner = Trainer(generate_results, env, agent, memory, name=name,
                     device=device, use_dyna=use_dyna, logger=log)

    runner.train_loop()


if __name__ == "__main__":
    main()

import torch
import logging
from envs import DMCSEnvironment
from memories import ReplayBuffer
from agents.mbrl import MBRL_DYNA_MNM_SAC
from networks.soft_actor import Actor
from networks.double_critic import DoubleQCritic
# from networks.distribution_Q import DoubleDistributionalQCritic
from networks.mbrl import Ensemble_World_Reward_GAN
from train_loops.trainer import Trainer
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
    use_bound = False
    on_policy = True

    # Environment settings.
    domain_name = "cartpole"
    task_name = "swingup"
    env = DMCSEnvironment(domain_name, task_name)
    action_dim = env.action_num
    state_dim = env.observation_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(device)
    # Algorithm settings.
    use_dyna = True
    alg = "dyna_mnm"
    if use_bound:
        bound = "_bound_"
    else:
        bound = "_norm_"
    if on_policy:
        policy = "on_policy"
    else:
        policy = "off_policy"
    name = alg + bound + policy

    num_models = 5
    capacity = 1000000

    actor = Actor(state_dim, action_dim)
    critic = DoubleQCritic(state_dim, action_dim)
    # critic = DoubleDistributionalQCritic(state_dim, action_dim)
    world_model = Ensemble_World_Reward_GAN(state_dim, action_dim, num_models)
    memory = ReplayBuffer((state_dim,), (action_dim,),
                          capacity, device)

    agent = MBRL_DYNA_MNM_SAC(actor, critic, world_model, device=device,
                              state_dim=state_dim,
                              action_dim=action_dim,
                              actor_lr=3e-4,
                              critic_lr=3e-4,
                              alpha_lr=3e-4,
                              gamma=0.99,
                              tau=0.005,
                              horizon=1,
                              sample_times=2,
                              on_policy=on_policy,
                              use_bound=use_bound)

    runner = Trainer(generate_results, env, agent, memory, name=name,
                     device=device, use_mbrl=use_dyna, logger=log)

    runner.train_loop()


if __name__ == "__main__":
    main()

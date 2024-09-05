import torch
import logging
from envs import OpenAIEnvrionment
from memories import MemoryBuffer
from agents.mbrl import DynaSAC_Reweight
from networks.mfrl.actor import Actor
from networks.mfrl.critic import Critic
from networks.mbrl import EnsembleWorldAndOneReward
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

    # Environment settings.
    env = OpenAIEnvrionment("Pendulum-v1")
    action_dim = env.action_num
    state_dim = env.observation_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Algorithm settings.
    alg = "pendulum_dyna_10_1_gt"
    name = alg
    num_models = 5

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    memory = MemoryBuffer()

    world_model = EnsembleWorldAndOneReward(observation_size=state_dim, num_actions=action_dim, num_models=num_models,
                                            device="cpu", lr=0.001)
    agent = DynaSAC_Reweight(actor, critic, world_model,
                             device=device,
                             action_num=action_dim,
                             actor_lr=3e-4,
                             critic_lr=3e-4,
                             alpha_lr=3e-4,
                             gamma=0.99,
                             tau=0.005,
                             horizon=1,
                             num_samples=20)

    runner = Trainer(env, agent, memory, name=name, device=device, logger=log)

    runner.train_loop()


if __name__ == "__main__":
    main()

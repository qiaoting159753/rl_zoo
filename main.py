from loops import MFRL_Trainer, MBRL_Trainer, World_Model_Trainer
from envs import DMCSEnvironment
from utils import set_seed


def main():
    """
    Step 1: Set seed
    Step 2: Set Environment: Real_Robot/DMCS
    Step 3: Set Training parameters.
    Step 4: Decide Loop: MFRL/MBRL/WM
    """
    # Random Seed
    seed = 10
    set_seed(seed)
    # Environment
    env = DMCSEnvironment("cheetah", "run")
    random_goal = False  # For real robot.
    action_dim = env.action_num
    state_dim = env.observation_space
    # Training
    G = 5
    model_G = 0.2
    batch_size = 256
    # Loop
    loop_name = "MFRL"
    # Agent
    device = "cpu"  # For mac training.
    agent_name = "SAC"

    if loop_name == "MFRL":
        trainer = MFRL_Trainer(env,
                               agent_name,
                               action_dim,
                               state_dim,
                               random_goal,
                               device,
                               G,
                               batch_size
                               )

    if loop_name == "MBRL":
        trainer = MBRL_Trainer(env,
                               agent_name,
                               action_dim,
                               state_dim,
                               random_goal,
                               device,
                               G,
                               batch_size
                               )

    if loop_name == "World_Model":
        trainer = World_Model_Trainer(env,
                               agent_name,
                               action_dim,
                               state_dim,
                               random_goal,
                               device,
                               G,
                               batch_size
                               )

    # Call training.
    trainer.train()


if __name__ == "__main__":
    main()



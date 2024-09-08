from loops import MFRL_Trainer, MBRL_Trainer, World_Model_Trainer
from envs import DMCSEnvironment
from utils import set_seed
import os


def main():
    """
    Step 1: Set seed
    Step 2: Set Environment: Real_Robot/DMCS
    Step 3: Set Training parameters.
    Step 4: Decide Loop: MFRL/MBRL/WM
    """
    seeds = [10, 25, 35, 45, 55]
    for seed in seeds:
        # Random Seed
        set_seed(seed)
        # Environment
        random_goal = True  # For real robot.
        env = DMCSEnvironment("cheetah", "run")
        env.set_seed(seed)
        # Training
        G = 1
        batch_size = 256
        episode_steps = 1000
        evaluate_interval = 1000
        maximum_steps = 1000000
        # Loop
        loop_name = "World_Model"
        # Agent
        # CHANGE THIS PART WHEN ON SCHOOL #
        directory = "statistics/"
        # directory = "/root/rl_zoo_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        device = "cpu"  # For mac training.
        agent_name = "Ensemble_Dyna_One_SAS_Reward"
        # MBRL
        model_G = 0.2
        horizon = 5
        branch_factor = 10
        trainer = None
        if loop_name == "MFRL":
            trainer = MFRL_Trainer(env=env,
                                   agent_name=agent_name,
                                   evaluate_interval=evaluate_interval,
                                   random_goal=random_goal,
                                   device=device,
                                   G=G,
                                   batch_size=batch_size,
                                   episode_steps=episode_steps,
                                   maximum_steps=maximum_steps,
                                   generate_results=True,
                                   seed=seed,
                                   directory=directory)
        if loop_name == "MBRL":
            trainer = MBRL_Trainer(env=env,
                                   agent_name=agent_name,
                                   random_goal=random_goal,
                                   device=device,
                                   G=G,
                                   model_G=model_G,
                                   batch_size=batch_size,
                                   horizon=horizon,
                                   branch_factor=branch_factor,
                                   episode_steps=episode_steps,
                                   maximum_steps=maximum_steps,
                                   seed=seed,
                                   directory=directory,
                                   generate_results=True,
                                   evaluate_interval=evaluate_interval)

        if loop_name == "World_Model":
            trainer = World_Model_Trainer(env=env,
                                          evaluate_interval=evaluate_interval,
                                          world_model_name=agent_name,
                                          random_goal=random_goal,
                                          device=device,
                                          on_policy=True,
                                          G=G,
                                          model_G=model_G,
                                          batch_size=batch_size,
                                          episode_steps=episode_steps,
                                          maximum_steps=maximum_steps,
                                          generate_results=True,
                                          seed=seed,
                                          directory=directory)
        # Call training.
        trainer.train()


if __name__ == "__main__":
    main()

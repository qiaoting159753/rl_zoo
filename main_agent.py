import logging
from loops import MFRL_Trainer, MBRL_Trainer, World_Model_Trainer
from envs import DMCSEnvironment
from utils import set_seed
import os
import json


def main():
    with open('configurations/bayesian_sgld.json', 'r') as file:
        data = json.load(file)
    logging.info(data)

    # EnV
    random_goal = data["random_goal"]
    env_domain = data["domain"]
    env_task = data["task"]
    # MBRL
    model_G = data["model_G"]
    horizon = data["horizon"]
    branch_factor = data["branch_factor"]
    # Training
    G = data["G"]
    loop_name = data["loop_name"]
    batch_size = data["batch_size"]
    episode_steps = data["episode_steps"]
    evaluate_interval = data["evaluate_interval"]
    maximum_steps = data["maximum_steps"]
    # Agents
    device = data["device"]  # For mac training.
    agent_name = data["agent_name"]
    seeds = data["seeds"]
    # Parameter tuning.
    sigmas = data["sigmas"]
    ratios = data["ratios"]

    for sigma in sigmas:
        for ratio in ratios:
            sub_directory = agent_name + "_" + env_domain + "_" + env_task + "_" + str(sigma) + "_" + str(ratio) + "/"
            # parent_dir = "/root/rl_zoo_data/"
            parent_dir = "statistics/"
            if not os.path.exists(parent_dir):
                os.mkdir(parent_dir)
            directory = os.path.join(parent_dir, sub_directory)
            if not os.path.exists(directory):
                os.mkdir(directory)

            for seed in seeds:
                # Random Seed
                set_seed(seed)
                # Environment
                env = DMCSEnvironment(env_domain, env_task)
                env.set_seed(seed)
                # Agent
                # CHANGE THIS PART WHEN ON SCHOOL #
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
                    on_policy = data["on_policy"]
                    trainer = World_Model_Trainer(env=env,
                                                  evaluate_interval=evaluate_interval,
                                                  world_model_name=agent_name,
                                                  random_goal=random_goal,
                                                  device=device,
                                                  on_policy=on_policy,
                                                  G=G,
                                                  model_G=model_G,
                                                  batch_size=batch_size,
                                                  episode_steps=episode_steps,
                                                  maximum_steps=maximum_steps,
                                                  generate_results=True,
                                                  seed=seed,
                                                  directory=directory,
                                                  ratio=ratio,
                                                  sigma=sigma)
                trainer.train()
                # try:
                #     trainer.train()
                # except Exception as e:
                #     logging.info("--------------------")
                #     logging.info(e)
                #     pass


if __name__ == "__main__":
    main()

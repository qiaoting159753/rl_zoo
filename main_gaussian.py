import logging
from loops import MFRL_Trainer, MBRL_Trainer, World_Model_Trainer
from envs import DMCSEnvironment
from utils import set_seed
import os
import json


def main():
    with open('configurations/gaussian_process.json', 'r') as file:
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
    noises = data["noises"]
    train_iters = data["train_iters"]

    for noise in noises:
        for train_iter in train_iters:
            sub_directory = agent_name + "_" + env_domain + "_" + env_task + "_" + str(noise) + "_" + str(train_iter)
            parent_dir = "/root/rl_zoo_data/"
            # parent_dir = "statistics/"
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
                                              ratio=train_iter,
                                              sigma=noise)
                try:
                    trainer.train(g_p=True)
                except Exception as e:
                    logging.info("--------------------")
                    logging.info(e)
                    pass


if __name__ == "__main__":
    main()

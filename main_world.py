import logging
from loops import World_Model_Trainer
from envs import DMCSEnvironment
from utils import set_seed
import os
import json


def main():

    with open('configurations/bayesian_la.json', 'r') as file:
        data = json.load(file)
    logging.info(data)

    # EnV
    env_domain = data["domain"]
    env_task = data["task"]
    random_goal = data["random_goal"]
    seeds = data["seeds"]
    # Agents
    device = data["device"]  # For mac training.
    agent_name = data["agent_name"]
    # MBRL
    model_G = data["model_G"]
    # Training
    G = data["G"]
    batch_size = data["batch_size"]
    episode_steps = data["episode_steps"]
    evaluate_interval = data["evaluate_interval"]
    maximum_steps = data["maximum_steps"]
    on_policy = data["on_policy"]
    # Parameter tuning.
    Parameter_A = data["Parameter_A"]
    Parameter_B = data["Parameter_B"]
    Parameter_C = data["Parameter_C"]
    flush = data["flush"]

    parent_dir = data["parent_direction"]
    # parent_dir = "statistics/"

    for parameter_a in Parameter_A:
        for parameter_b in Parameter_B:
            for parameter_c in Parameter_C:
                sub_directory = agent_name + "_" + env_domain + "_" + env_task + "/"
                if not os.path.exists(parent_dir):
                    os.mkdir(parent_dir)

                directory = os.path.join(parent_dir, sub_directory)
                if not os.path.exists(directory):
                    os.mkdir(directory)
                directory = directory + str(parameter_a) + "_" + str(parameter_b) + "_" + str(parameter_c) + "_"

                for seed in seeds:
                    # Random Seed
                    set_seed(seed)
                    # Environment
                    env = DMCSEnvironment(env_domain, env_task)
                    env.set_seed(seed)
                    # Agent
                    # CHANGE THIS PART WHEN ON SCHOOL #
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
                                                  parameter_a=parameter_a,
                                                  parameter_b=parameter_b,
                                                  parameter_c=parameter_c)
                    try:
                        trainer.train(flush=flush)
                    except Exception as e:
                        logging.info("--------------------")
                        logging.info(e)
                        pass

if __name__ == "__main__":
    main()

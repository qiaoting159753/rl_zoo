import logging
import shutil
from rl_zoo.loops import World_Model_Trainer, MBRL_Trainer
from rl_zoo.envs import DMCSEnvironment, OpenAIEnvrionment
from rl_zoo.utils import set_seed
import os
import json


def main():
    curr_path = os.getcwd()
    alg_name = "mbrl_ir_dyna_sac.json"
    config_file = curr_path + '/rl_zoo/configurations/' + alg_name
    with open(config_file, 'r') as file:
        data = json.load(file)
    logging.info(data)
    # EnV
    random_goal = data["random_goal"]
    seeds = data["seeds"]
    # Agents
    device = data["device"]  # For mac training.
    agent_name = data["agent_name"]
    # MBRL
    model_G = data["model_G"]
    # Training
    G = data["G"]
    loop_name = data["loop_name"]
    batch_size = data["batch_size"]
    episode_steps = data["episode_steps"]
    evaluate_interval = data["evaluate_interval"]
    maximum_steps = data["maximum_steps"]
    flush = data["flush"]
    on_policy = data["on_policy"]
    # Parameter tuning.
    Parameter_A = data["Parameter_A"]
    Parameter_B = data["Parameter_B"]
    Parameter_C = data["Parameter_C"]
    # Reward
    gripper = data["gripper"]
    sas = data["sas"]
    prob_rwd = data["prob_rwd"]
    train_both = data["train_both"]
    train_reward = data['train_reward']
    # Switch
    parent_dir = curr_path + "/statistics/"
    agents = ["ball_in_cup", "cheetah", "hopper", "Pusher-v5", "Ant-v5"]
    tasks = ["catch", "run", "hop", "", ""]
    for i in range(len(agents)):
        env_domain = agents[i]
        env_task = tasks[i]
        sub_directory = agent_name + "_" + env_domain + "_" + env_task + "/"
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        directory = os.path.join(parent_dir, sub_directory)
        if not os.path.exists(directory):
            os.mkdir(directory)
        shutil.copyfile(config_file, directory + alg_name)

        for parameter_a in Parameter_A:
            for parameter_b in Parameter_B:
                for parameter_c in Parameter_C:
                    direct_param = directory + str(parameter_a) + "_" + str(parameter_b) + "_" + str(parameter_c) + "_"
                    for seed in seeds:
                        # Random Seed
                        set_seed(seed)
                        # Environment
                        if i > 1:
                            env = OpenAIEnvrionment(env_domain, param=True)
                        else:
                            env = DMCSEnvironment(env_domain, env_task)
                        env.set_seed(seed)
                        # Agent
                        # CHANGE THIS PART WHEN ON SCHOOL #
                        if loop_name == "World_Model":
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
                                                          directory=direct_param,
                                                          parameter_a=parameter_a,
                                                          parameter_b=parameter_b,
                                                          parameter_c=parameter_c,
                                                          sas=sas,
                                                          train_both=train_both,
                                                          train_reward=train_reward,
                                                          prob_rwd=prob_rwd)
                        if loop_name == "MBRL":
                            trainer = MBRL_Trainer(env=env,
                                                   evaluate_interval=evaluate_interval,
                                                   mbrl_agent_name=agent_name,
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
                                                   directory=direct_param,
                                                   parameter_a=parameter_a,
                                                   parameter_b=parameter_b,
                                                   parameter_c=parameter_c,
                                                   sas=sas,
                                                   train_both=train_both,
                                                   train_reward=train_reward,
                                                   prob_rwd=prob_rwd,
                                                   gripper=gripper)
                        trainer.train(flush=flush)


if __name__ == "__main__":
    main()

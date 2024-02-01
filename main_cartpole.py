import logging
logging.basicConfig(level=logging.INFO)
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import random
from collections import namedtuple
import torch
from prev_data.imp_cartpole import CartPoleEnv
from components import ReplayMemory
from prev_data.agent_dqn import Agent


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_agent(env, agent, num_eval=5):
    total_rewards = 0
    for i in range(num_eval):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for j in range(500):
            # Act purly based on agent.
            with torch.no_grad():
                action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_rewards += reward
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            state = next_state
            if done:
                break
    avg_rewards = total_rewards / num_eval
    logging.info(msg=f'Evaluation: {avg_rewards}')


def train_agent(env, memory, agent, batch_size=128, max_epi_steps=500, num_training=10):
    for i_episode in range(num_training):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        train_total_rewards = 0
        for j in range(max_epi_steps):
            # Action tensor
            eps_threshold = 0.1
            sample = random.random()
            if sample > eps_threshold:
                with torch.no_grad():
                    action = agent.select_action(state)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            train_total_rewards += reward

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            memory.push(state, action, next_state, reward)

            if len(memory) > batch_size:
                for k in range(10):
                    transitions = memory.sample(batch_size=batch_size)
                    statistics = memory.get_statistic()
                    agent.train_world_model(statistics=statistics, transitions=transitions)

                for l in range(1):
                    transitions = memory.sample(batch_size=batch_size)
                    agent.optimize_model_multi(transitions)

            # Move to the next state
            state = next_state
            if done:
                # if len(memory) > batch_size:
                #     agent.world_model.evaluate_world_model_critic(env, agent.target_net, device)

                #     agent.world_model.evaluate_world_model(env)
                # logging.info(msg=f'Training: {train_total_rewards}')
                break


if __name__ == '__main__':
    BATCH_SIZE = 128
    memory = ReplayMemory(10000)
    env = CartPoleEnv()
    agent = Agent(env=env, n_obs=env.observation_space.shape[0], n_actions=env.action_space.n, device=device,
                  batch_size=BATCH_SIZE)

    with logging_redirect_tqdm():
        for i in trange(200):
            train_agent(env, memory, agent, BATCH_SIZE)
            evaluate_agent(env, agent)

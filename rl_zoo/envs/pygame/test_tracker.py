from rl_zoo.envs import PusherEnv
from rl_zoo.agents.mbrl import MBRL_SAC
import torch

batch_size = 256
PATH = "zTracker-v4_actor_params_2_2_sac.pth"

env = PusherEnv(render_mode="human")
agent = MBRL_SAC(env=env, state_dim=env.observation_space.shape[0],
                                    action_dim=env.action_space.shape[0],
                                    batch_size=batch_size)

agent.actor.load_state_dict(torch.load(PATH, map_location="cpu"))

for i in range(100):
    state, _ = env.reset()
    for j in range(1000):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        state = next_state






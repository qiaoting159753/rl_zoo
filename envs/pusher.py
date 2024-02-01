import math
import random
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import time

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}


class PusherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "pusher.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self.change_direction_x = False
        self.change_direction_y = False
        self.velocity_x = (random.random() - 0.5) * 0.4
        self.velocity_y = (random.random() - 0.5) * 0.4
        self.location_x = 0
        self.location_y = 0
        self.prev_time = 0.0

    def compute_next_goal(self):
        # Update goal position
        observation = self._get_obs()  # [-0.35, 0.25]
        qpos = np.zeros(11)
        qvel = np.zeros(11)  # Rest 4 are cylinder and goal velocities.
        qpos[-4:-2] = self.cylinder_pos  # Cylinder Position
        qvel[0:7] = observation[7:14]  # Joint velocity
        qpos[0:7] = observation[0:7]  # Joint Position
        if self.change_direction_x:
            self.velocity_x *= -1
            self.change_direction_x = False
        if self.change_direction_y:
            self.velocity_y *= -1
            self.change_direction_y = False
        # Compute the next location of the auruco code.
        time_interval = time.time() - self.prev_time
        next_x = self.location_x + time_interval * self.velocity_x
        next_y = self.location_y + time_interval * self.velocity_y
        if next_y > 0.2: # in out
            next_y = 0.2
            self.change_direction_y = True
        if next_y < -0.3:
            next_y = -0.3
            self.change_direction_y = True
        if next_x > 0.1:
            next_x = 0.1
            self.change_direction_x = True
        if next_x < -0.9:
            next_x = -0.9
            self.change_direction_x = True
        self.location_x = next_x
        self.location_y = next_y
        qpos[-2:] = [next_y, next_x]
        self.prev_time = time.time()
        self.set_state(qpos, qvel)

    def step(self, a):
        self.compute_next_goal()
        goal_pos = self.get_body_com("goal")
        goal_pos[2] += 0.063
        goal_pos[0] -= 0.046
        vec_1 = goal_pos - self.get_body_com("tips_arm")
        # vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        # vec_3 = self.get_body_com("tips_arm") - self.get_body_com("goal")
        reward_reachring = -np.linalg.norm(vec_1)
        reward_dist = 0
        # reward_near = -np.linalg.norm(vec_1)
        # reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(a).sum()
        reward = reward_reachring # + 0.1 * reward_ctrl  # + 0.5 * reward_near
        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()
        ob = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset_model(self):
        self.velocity_x = (random.random() - 0.5) * 0.3 + 0.2
        self.velocity_y = (random.random() - 0.5) * 0.3 + 0.2
        # [joint_angle0,ja_1,ja_2,ja_3,ja_4,ja_5,ja_6,cylinder_x,cylinder_y,goal_x,goal_y]
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([1.0, 1.0])
        qpos[-4:-2] = self.cylinder_pos
        # qpos[-2:] = [0.0827, 0.0731]  # self.goal_pos # [0,0] = [0.45, -0.05, -0.323]
        qpos[-2:] = [0.0, 0.0]
        # qpos[0] = 0.91
        # qpos[1] = 0.37
        # qpos[3] = 0.00
        self.set_state(qpos, qvel)
        # vec_1 = self.get_body_com("goal") - self.get_body_com("tips_arm")
        # print("-------------------------")
        # goal_pos = self.get_body_com("goal")
        # tip_pos = self.get_body_com("tips_arm")
        # print(goal_pos[2] + 0.063)
        # print(tip_pos[2])
        # (goal_pos[0] - 0.046 - tip_pos) ** 2
        # print(self.get_body_com("tips_arm"))
        # reward_reachring = -np.linalg.norm(vec_1)
        # print(reward_reachring)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:7],
                self.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )


if __name__ == "__main__":
    env = PusherEnv(render_mode="human")
    state, _ = env.reset()
    goal_position = state[-6:]

    while True:
        env.reset()
        for i in range(1000):
            action = np.zeros(7) # env.action_space.sample()
            next_state, reward, terminate, truncted, info = env.step(action)

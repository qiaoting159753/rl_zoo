from functools import cached_property
import cv2
import numpy as np
from dm_control import suite
import gymnasium as gym
from gymnasium import spaces
import logging
from utils.reward_functions import get_dmcs_reacher_reward, get_dmcs_finger_reward, get_dmcs_fish_reward
from utils.reward_functions import get_openai_hopper_reward, get_openai_walker_reward, get_openai_halfcheetah_reward


class Environment:
    """
    Abstract of Environment contains name and state, action size.
    """

    def __init__(self, domain_name, task_name):
        self.domain = domain_name
        self.task = task_name

    def observation_space(self) -> int:
        """
        Fake State dimension
        :return:
        """
        return 1000

    def action_num(self) -> int:
        """
        Fake action dimension
        :return:
        """
        return 1000


class DMCSEnvironment(Environment):
    """
    Deepmind Control Suite.

    """

    def __init__(self, domain, task) -> None:
        super().__init__(domain_name=domain, task_name=task)
        self.task = task
        logging.info(f"Training on Domain {domain}")
        self.domain = domain
        self.env = suite.load(self.domain, self.task)

    @cached_property
    def min_action_value(self) -> float:
        return self.env.action_spec().minimum[0]

    @cached_property
    def max_action_value(self) -> float:
        return self.env.action_spec().maximum[0]

    @cached_property
    def observation_space(self) -> int:
        time_step = self.env.reset()
        # e.g. position, orientation, joint_angles
        observation = np.hstack(list(time_step.observation.values()))
        return len(observation)

    @cached_property
    def action_num(self) -> int:
        return self.env.action_spec().shape[0]

    def sample_action(self) -> int:
        return np.random.uniform(
            self.min_action_value, self.max_action_value, size=self.action_num
        )

    def set_seed(self, seed: int) -> None:
        self.env = suite.load(self.domain, self.task, task_kwargs={"random": seed})

    def reset(self) -> np.ndarray:
        time_step = self.env.reset()
        observation = np.hstack(
            list(time_step.observation.values())
        )  # # e.g. position, orientation, joint_angles
        return observation

    def step(self, action: int) -> tuple:
        time_step = self.env.step(action)
        state, reward, done = (
            np.hstack(list(time_step.observation.values())),
            time_step.reward,
            time_step.last(),
        )
        # for consistency with open ai gym just add false for truncated
        return state, reward, done, False

    def grab_frame(self, height=480, width=640, camera_id=0) -> np.ndarray:
        frame = self.env.physics.render(camera_id=camera_id, height=height, width=width)
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("original_" + self.domain + "_" + self.task + ".png", frame)
        return frame

    def get_gt_reward(self, state, action, next_state):
        if self.domain == "reacher":
            return get_dmcs_reacher_reward(self, next_state)
        elif self.domain == "finger":
            return get_dmcs_finger_reward(self, next_state)
        elif self.domain == "fish":
            return get_dmcs_fish_reward(self, next_state)
        else:
            raise NotImplementedError("Ground Truth Reward function is not implemented!")

class OpenAIEnvrionment:
    def __init__(self, task_name, param) -> None:
        self.task_name = task_name
        self.param = param
        self.env = gym.make(task_name, render_mode="rgb_array", exclude_current_positions_from_observation=param)

    @cached_property
    def max_action_value(self) -> float:
        return self.env.action_space.high[0]

    @cached_property
    def min_action_value(self) -> float:
        return self.env.action_space.low[0]

    @cached_property
    def observation_space(self) -> int:
        return self.env.observation_space.shape[0]

    @cached_property
    def action_num(self) -> int:
        if isinstance(self.env.action_space, spaces.Box):
            action_num = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            action_num = self.env.action_space.n
        else:
            raise ValueError(
                f"Unhandled action space type: {type(self.env.action_space)}"
            )
        return action_num

    def sample_action(self) -> int:
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        _, _ = self.env.reset(seed=seed)
        # Note issues: https://github.com/rail-berkeley/softlearning/issues/75
        self.env.action_space.seed(seed)

    def reset(self) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> tuple:
        state, reward, done, truncated, _ = self.env.step(action)
        return state, reward, done, truncated

    def grab_frame(self, height=240, width=300) -> np.ndarray:
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_gt_reward(self, state, action, next_state):
        if self.task_name == "HalfCheetah-v5" and not self.param:
            return get_openai_halfcheetah_reward(state, action, next_state)
        elif self.task_name == "Hopper-v5" and not self.param:
            return get_openai_hopper_reward(state, action, next_state)
        elif self.task_name == "Walker2d-v5" and not self.param:
            return get_openai_walker_reward(state, action, next_state)
        else:
            raise NotImplementedError("Ground Truth Reward function is not implemented!")

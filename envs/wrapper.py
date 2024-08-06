from functools import cached_property
import cv2
import numpy as np
from dm_control import suite
import gymnasium as gym
from gymnasium import spaces


class DMCSEnvironment:
    """

    """
    def __init__(self, domain_name, task_name, seed=10):
        super().__init__()
        self.domain = domain_name
        self.task = task_name
        self.env = suite.load(domain_name, task_name, task_kwargs={"random": seed})

    @cached_property
    def min_action_value(self) -> float:
        """

        :return:
        """
        return self.env.action_spec().minimum[0]

    @cached_property
    def max_action_value(self) -> float:
        """

        :return:
        """
        return self.env.action_spec().maximum[0]

    @cached_property
    def observation_space(self) -> int:
        """

        :return:
        """
        time_step = self.env.reset()
        # e.g. position, orientation, joint_angles
        observation = np.hstack(list(time_step.observation.values()))
        return len(observation)

    @cached_property
    def action_num(self) -> int:
        """

        :return:
        """
        return self.env.action_spec().shape[0]

    def set_seed(self, seed: int) -> None:
        """

        :param seed:
        """
        self.env = suite.load(self.domain, self.task, task_kwargs={"random": seed})

    def reset(self) -> np.ndarray:
        """

        :return:
        """
        time_step = self.env.reset()
        observation = np.hstack(
            list(time_step.observation.values())
        )  # # e.g. position, orientation, joint_angles
        return observation

    def step(self, action: int) -> tuple:
        """

        :param action:
        :return:
        """
        time_step = self.env.step(action)
        state, reward, done = (
            np.hstack(list(time_step.observation.values())),
            time_step.reward,
            time_step.last(),
        )
        # for consistency with open ai gym just add false for truncated
        return state, reward, done, False

    def grab_frame(self, height=240, width=300, camera_id=0) -> np.ndarray:
        """

        :param height:
        :param width:
        :param camera_id:
        :return:
        """
        frame = self.env.physics.render(camera_id=camera_id, height=height, width=width)
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


class OpenAIEnvrionment:
    """
    OpenAi Gym
    """
    def __init__(self, task_name) -> None:
        self.env = gym.make(task_name, render_mode="rgb_array")

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
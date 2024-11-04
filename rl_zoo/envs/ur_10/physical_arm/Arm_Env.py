import numpy as np
from arm.Arm import Arm
from arm.forward_kinematics import Forward_Kinematics
import random
import math


def _value_to_radii(values):
    return (math.pi/180) * (values * 0.292969)


class Arm_Env:
    """
    The Environment interface.

    One Robotic arm.
    One IMU sensor.
    One Camera Sensor.
    """
    def __init__(self):
        # Create the arm and ping it first time.
        self.arm = Arm() # Create and lock.
        self.kinematic = Forward_Kinematics()

    def reset(self):
        # Stop moving.
        self.arm.stop()
        # Execute the home sequence.
        self.arm.go_home()
        # Get State
        return self._get_state()

    def sample_action(self, scale=0.05):
        """
        Random create actions that only 0.1 of the scale of each joint.

        :param scale:
        :return:
        """
        rand_actions = []
        for i in range(len(self.arm.position_mins)):
            # Move to higher or lower.
            numer = random.uniform(-0.5, 0.5) * self.arm.scales[i] * scale
            numer = int(numer)
            rand_actions.append(numer)
        return np.array(rand_actions)

    def step(self, action, time_out=5):
        for i in range(5):
            continue
        # See if it is finished moving
        # # Get current state
        # positions = self.arm.get_positions()
        # # Add current + action
        # target = positions + action
        # # Move to the target.
        # self.arm.bulk_move(target)

    def _get_state(self):
        print("forming states")
        values = self.arm.get_positions()
        radii = _value_to_radii(values)
        end_pos, end_ori = self.kinematic.forward_kinematics(radii)
        return radii

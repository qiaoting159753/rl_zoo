from scipy.spatial.transform import Rotation
import numpy as np
import math
import random
import torch
from .env_arm_utils import a, alph, d, matrix_to_quaternion
from .ur10_forward_kinematics import Forward_Kinematics
import pyquaternion
from pyquaternion import Quaternion


class UR10_Kinematic_Env:
    """
    I used a forward kinematics to create a fake environment for robotic arm reaching and tracking task.

    Always, changing the reward function to be sensitive to small d istance helps!
    For instance, in most cases, reward = dist, dist is 0-120. However, we want to emphasize the small distance
    for the accuracy sake.

    :param full_joint_dim: Number of joints in this robotic arm in total.
    :param full_joint_dim:
    """

    def __init__(self):
        # For goal Definition.
        self.tolerate_range = 0.3  # in cm
        self.quaternion_tolerate_range = 0.0007  # 0.0007: 3 deg, 0.0019: 5 deg.
        # Goals
        self.goal = np.array([30, 0.0001, 0.0001])
        goal_rot = Rotation.from_euler("xyz", np.array([0.414, math.pi, 1.5]))
        self.quaternion_goal = np.roll(goal_rot.as_quat(canonical=True), shift=1)
        # 3: 0.5 cm, 9: 1.56
        self.step_angle = 9
        self.angle = 0  # Unit: degree. Increase angle by a few degrees.
        self.current_joint_position = np.zeros(6) + 0.000001
        self.full_joint_dim = 6
        self.fk = Forward_Kinematics()

    def sample_action(self, row, column=6):
        """
        Sample a specific size of actions. [Batch_size, Number_actions]
        :param row:
        :param column:
        :return:
        """
        # random between -1, 1
        return (np.random.rand(row, column) - 0.5) * 2

    def reset(self, random_goal=False, return_home=False):
        """
        Randomize where the goal is / set a specific goal / Fixed Goal/
        The robotic arm will be returned home / not to.

        :param random_goal:
        :param return_home:
        :return:
        """
        if random_goal:
            # 2D Random
            not_inside = True
            while not_inside:
                x = (random.uniform(0, 1)) * 35
                y = (random.uniform(0, 1) - 0.5) * 70
                z = 0.001
                dist = x ** 2 + y ** 2 + z ** 2
                dist = math.sqrt(dist)
                if dist < 50:
                    not_inside = False
                    self.goal = np.array([x, y, z])
        else:
            # Circle: Origin: 20, 0, 0
            # Diameter: 10:
            # Start: 30, 0, 0
            # self.goal = np.array([25, 0, 25])
            radii = ((self.angle + 1) % 180) / 180 * math.pi
            x = 10 * math.sin(radii) + 20  # Diameter and Offset
            y = 10 * math.cos(radii)
            z = 0.001
            self.goal = np.array([x, y, z])
            self.angle += self.step_angle
        if return_home:
            self.current_joint_position = np.zeros(self.full_joint_dim) + 0.00001
        state = self._get_observation()
        return state

    def _get_observation(self):
        """
        Full size state, (gx, gy, gz,
                          end_x, end_y, end_z,
                          qgx, qgy, qgz,qgw,
                          qx, qy, qz, qw,
                          j0, j1, j2, j3, j4, j5)
        :return:
        """
        self.current_end_factor = self.fk.forward_kinematics(self.current_joint_position)
        trans = Rotation.from_matrix(self.current_end_factor[0:3, 0:3])
        quaternion = np.roll(trans.as_quat(canonical=True), shift=1)
        positions = np.array(
            [self.current_end_factor[0, 3], self.current_end_factor[1, 3], self.current_end_factor[2, 3]])
        obs = np.concatenate((self.goal, positions, self.quaternion_goal, quaternion, self.current_joint_position))
        return obs

    def _reward_1(self, state):
        dist = (state[0] - state[3]) ** 2 + (state[1] - state[4]) ** 2 + (state[2] - state[5]) ** 2
        dist = math.sqrt(dist)
        reward = -5 * (np.tanh(dist / 50))
        return reward, dist

    def _reward_2(self, state):

        a = state[6]
        b = state[7]
        c = state[8]
        d = state[9]

        e = state[10]
        f = state[11]
        g = state[12]
        h = state[13]

        q2m = np.matrix([[e, -1 * f, -1 * g, -1 * h],
                         [f, e, -1 * h, g],
                         [g, h, e, -1 * f],
                         [h, -1 * g, f, e]])
        q1i = np.matrix([[a], [-1 * b], [-1 * c], [-1 * d]])

        quat_dis = np.matmul(q2m, q1i)
        diff_rot = Rotation.from_quat(np.squeeze(np.asarray(quat_dis)))
        diff_euler = diff_rot.as_euler('zxy')
        quaternion_dist = math.sqrt(diff_euler[0] ** 2 + diff_euler[1] ** 2 + diff_euler[2] ** 2)

        # quaternion_dist = ((state[6] * state[10]) + (state[7] * state[11]) + (state[8] * state[12]) + (state[9] * state[13])) ** 2
        # q_1 = Quaternion(np.array([state[6], state[7], state[8], state[9]]))
        # q_2 = Quaternion(np.array([state[10], state[11], state[12], state[13]]))
        # quaternion_dist = Quaternion.absolute_distance(q_1, q_2)

        # angle_p = Rotation.from_euler("zyx", [30/180 * math.pi,  5/180 * math.pi, 15/180 * math.pi])
        # angle_q = Rotation.from_euler("zyx", [31/180 * math.pi, 11/180 * math.pi, 15/180 * math.pi])
        # quat_p = np.roll(angle_p.as_quat(canonical=True), shift=1)
        # quat_q = np.roll(angle_q.as_quat(canonical=True), shift=1)
        # state[6:10] = quat_p
        # state[10:14] = quat_q

        # p = np.array([state[10], state[11], state[12], state[13]])
        # c_q = np.array([state[6], -1 * state[7], -1 * state[8], -1 * state[9]])
        # # r
        # z_0 = abs(p[0] * c_q[0] - p[1] * c_q[1] - p[2] * c_q[2] - p[3] * c_q[3])
        # # i
        # z_1 = abs(p[0] * c_q[1] + p[1] * c_q[0] + p[2] * c_q[3] - p[3] * c_q[2])
        # # j
        # z_2 = abs(p[0] * c_q[2] + p[2] * c_q[0] - p[1] * c_q[3] + p[3] * c_q[1])
        # # k
        # z_3 = abs( p[0] * c_q[3] + p[3] * c_q[0] + p[1] * c_q[2] - p[2] * c_q[1])
        # print("-------------------")
        # print(z_0)
        # quaternion_dist = (2 * math.acos(z_0))
        # print(2 * math.acos(z_1))
        # print(2 * math.acos(z_2))
        # print(2 * math.acos(z_3))

        # (p[0] + p[1]i + p[2]j + p[3]k) * (c_q[0] + c_q[1]i + c_q[2]j + c_q[3]k) =
        # p[0] * c_q[0] +
        # p[0] * c_q[1]i +
        # p[0] * c_q[2]j +
        # p[0] * c_q[3]k +

        # p[1]i * c_q[0] +
        # p[1]i * c_q[1]i +
        # p[1]i * c_q[2]j + ij = k
        # p[1]i * c_q[3]k + ik = -j

        # p[2]j * c_q[0] +
        # p[2]j * c_q[1]i + ji = -k
        # p[2]j * c_q[2]j +
        # p[2]j * c_q[3]k + jk = i

        # p[3]k * c_q[0] +
        # p[3]k * c_q[1]i + ki = j
        # p[3]k * c_q[2]j + kj = -i
        # p[3]k * c_q[3]k +

        # Quaternion to Euler.
        # zxy : 0.52, 0.50, 0.52, 0.46, 0.50
        r1 = Rotation.from_quat(np.array([state[6], state[7], state[8], state[9]]))
        r2 = Rotation.from_quat(np.array([state[10], state[11], state[12], state[13]]))
        angles_r1 = r1.as_euler('zxy')
        angles_r2 = r2.as_euler('zxy')
        angle_diff = angles_r1 - angles_r2
        angle_dist = (angle_diff[0]) ** 2 + (angle_diff[1]) ** 2 + (angle_diff[2]) ** 2
        angle_dist = math.sqrt(angle_dist)
        return - 5 * (np.tanh(1 - quaternion_dist)), quaternion_dist, angle_dist

    def _get_reward(self, prev_state, action, state):
        """
        How to define the reward? For now, the simplist: -1 * distance to the goal.

        :param prev_state: The state before this action.
        :param action: Action that is executed.
        :param state: The state after this action.
        :return:
        """
        # Distance rewards.
        euclidean_reward, dist = self._reward_1(state)
        angular_reward, quat_dist, euler_dist = self._reward_2(state)
        reward = euclidean_reward + angular_reward
        return reward, euclidean_reward, angular_reward, dist, quat_dist, euler_dist

    def step(self, action):
        """
        Take in a action, add it to the current state, clip it (-pi, +pi), forward kinematic it.

        :param action: Full size action. Have to be the same as the predefined. [full_joint_dim,]
        :return: State:
        """
        assert action.shape[0] == self.full_joint_dim
        # prev_observation = self._get_observation()
        self.current_joint_position += action
        # Clip to range [-pi, pi]
        self.current_joint_position = np.clip(self.current_joint_position, a_min=-math.pi, a_max=math.pi)
        observation = self._get_observation()
        # Fake a prev observation for now.
        reward, _, _, dist, quaternion_dist, euler_dist = self._get_reward(observation, action, observation)
        done = False
        info = {"quat_dist": quaternion_dist, "euler_dist": euler_dist}
        # If very close the the desired goal.
        if (dist < self.tolerate_range) and (quaternion_dist < self.quaternion_tolerate_range):
            done = True
            reward = 10  # 1.0 / dist
        return observation, reward, done, dist, info

    def tensor_query(self, state_tensor, action_tensor):
        """
        This function suppose to be the same as the step function but take partial state, and partial action as input.

        :param state_tensor: 6 (goal + end_posiiton) + do_joint.
        :param action_tensor: do_joint.
        :return:
        """
        # Pre-process convert to full length
        full_actions = torch.zeros((state_tensor.shape[0], self.full_joint_dim))
        full_states = torch.zeros((state_tensor.shape[0], self.full_joint_dim))
        # 0    1   2  3  4  5    6    7    8    9  10  11  12  13  14  15  16  17  18  19
        # gx, gy, gz, x, y, z, gqx, gqy, gqz, gqw, qx, qy, qz, qw, j0, j1, j2, j3, j4, j5.
        partial_joints_states = state_tensor[:, 14:]
        full_actions[:, 0:action_tensor.shape[1]] = action_tensor
        full_states[:, 0:partial_joints_states.shape[1]] = partial_joints_states
        # Add and clip.
        full_states += full_actions
        joints_tensor = torch.clip(full_states, min=-math.pi, max=math.pi)
        ############    This part is the fullly Forward Kinematics.    ##########
        batch_size = joints_tensor.shape[0]
        num_joints = joints_tensor.shape[1]
        T_a_s = []
        T_d_s = []
        Rxa_s = []
        Rzt_s = []
        joint_angle_coss = torch.cos(joints_tensor)
        joint_angle_sins = torch.sin(joints_tensor)
        for i in range(num_joints):
            T_a = torch.FloatTensor(np.identity(4))
            T_a[0, 3] = a[0, i]
            T_a = T_a.unsqueeze(dim=0)
            T_a = torch.repeat_interleave(T_a, batch_size, dim=0)
            T_a_s.append(T_a)
            T_d = torch.FloatTensor(np.identity(4))
            T_d[2, 3] = d[0, i]
            T_d = T_d.unsqueeze(dim=0)
            T_d = torch.repeat_interleave(T_d, batch_size, dim=0)
            T_d_s.append(T_d)
            Rxa = torch.FloatTensor(np.identity(4)).unsqueeze(dim=0)
            Rxa = torch.repeat_interleave(Rxa, repeats=batch_size, dim=0)
            Rxa[:, 1, 1] = math.cos(alph[0, i])
            Rxa[:, 1, 2] = -1 * math.sin(alph[0, i])
            Rxa[:, 2, 1] = math.sin(alph[0, i])
            Rxa[:, 2, 2] = math.cos(alph[0, i])
            Rxa_s.append(Rxa)
            Rzt = torch.FloatTensor(np.identity(4)).unsqueeze(dim=0)
            Rzt = torch.repeat_interleave(Rzt, repeats=batch_size, dim=0)
            Rzt[:, 0, 0] = joint_angle_coss[:, i]
            Rzt[:, 0, 1] = -1 * joint_angle_sins[:, i]
            Rzt[:, 1, 0] = joint_angle_sins[:, i]
            Rzt[:, 1, 1] = joint_angle_coss[:, i]
            Rzt_s.append(Rzt)
        T_a_s = torch.stack(T_a_s)
        T_d_s = torch.stack(T_d_s)
        Rxa_s = torch.stack(Rxa_s)
        Rzt_s = torch.stack(Rzt_s)
        A_i = torch.matmul(torch.matmul(torch.matmul(T_d_s, Rzt_s), T_a_s), Rxa_s)
        together_mat = torch.FloatTensor(np.identity(4)).unsqueeze(0)
        together_mat = torch.repeat_interleave(together_mat, batch_size, dim=0)
        # Forward kinematics.
        for i in range(num_joints):
            together_mat = torch.matmul(together_mat, A_i[i])
        # Distance to goal: Now - Goal.
        quaternion = matrix_to_quaternion(together_mat[:, 0:3, 0:3])
        quaternion_losses = quaternion[:, 0:4] * state_tensor[:, 6:10]
        quaternion_losses = torch.sum(quaternion_losses, dim=1)
        quaternion_losses = torch.pow(quaternion_losses, exponent=2)

        losses = torch.square(together_mat[:, 0:3, 3] - state_tensor[:, :3])
        losses = torch.sum(losses, dim=1)
        losses = torch.sqrt(losses)

        # Reward = -1 * dist
        rewards = -5 * (torch.tanh(losses / 50)) - 5 * (torch.tanh(quaternion_losses))
        rewards = rewards.unsqueeze(dim=1)
        # Done Prediction
        dones = torch.logical_and((losses < self.tolerate_range), (quaternion_losses < self.quaternion_tolerate_range))
        # dones = (losses < self.tolerate_range)
        # inv_dist = (1.0 / losses).unsqueeze(dim=1)
        rewards[dones] = 10.0  # inv_dist[dones]
        # Formation the states, The next state here should be partial next state. state_tensor
        next_state = torch.zeros(state_tensor.shape) + 0.00001
        next_state[:, 0:3] = state_tensor[:, 0:3]
        next_state[:, 3:6] = together_mat[:, 0:3, 3]

        next_state[:, 6:10] = state_tensor[:, 6:10]
        next_state[:, 10:14] = quaternion[:, 0:4]

        # joint_tensor: full
        next_state[:, 14:] = joints_tensor[:, 0:partial_joints_states.shape[1]]
        # Paritial = Partial
        assert (state_tensor.shape[1] - 14) == partial_joints_states.shape[1]
        return next_state, rewards, dones

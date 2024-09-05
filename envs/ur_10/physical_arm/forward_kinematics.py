import numpy as np
import math
from scipy.spatial.transform import Rotation as R


class Forward_Kinematics:
    def __init__(self):
        print("Low-cost robot")
        self.d_s =     [15.6,             0.0,        6.0,       0.0,        4.4, 0.0, 0.0]
        self.a_s =     [0.0,             10.6,        0.0,       9.6,        0.0, 8.0, 10.0]
        self.alpha_s = [-math.pi/2, math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, 0.0, 0.0]

        gamma = math.pi / 2
        self.rot_y = np.matrix(np.array([[math.cos(gamma), 0.0, math.sin(gamma), 0.0],
                                         [0.0, 1.0, 0.0, 0.0],
                                         [-math.sin(gamma), 0.0, math.cos(gamma), 0.0],
                                         [0.0, 0.0, 0.0, 1.0]]))
        self.rot_y_2 = np.matrix(np.array([[math.cos(-gamma), 0.0, math.sin(-gamma), 0.0],
                                         [0.0, 1.0, 0.0, 0.0],
                                         [-math.sin(-gamma), 0.0, math.cos(-gamma), 0.0],
                                         [0.0, 0.0, 0.0, 1.0]]))

        phi = math.pi / 2
        self.rot_x = np.matrix(np.array([[1.0, 0.0, 0.0, 0.0],
                                         [0.0, math.cos(phi), -math.sin(phi), 0.0],
                                         [0.0, math.sin(phi), math.cos(phi), 0.0],
                                         [0.0, 0.0, 0.0, 1.0]]))

        self.corr_1 = np.matrix(np.array([[0.0, 0.0, -1.0, 0.0],
                                          [-1.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 1.0]]))

        self.corr_3 = np.identity(4)
        self.corr_3[1, 3] = -10.0

    def forward_kinematics(self, thetas):

        thetas[3] = -1 * thetas[3]
        thetas[6] = -1 * thetas[6]

        A_01 = self.one_link(thetas[0], self.d_s[0], self.a_s[0], self.alpha_s[0])
        A_12 = self.one_link(thetas[1] - math.pi/2, self.d_s[1], self.a_s[1], self.alpha_s[1])
        A_23 = self.one_link(thetas[2], self.d_s[2], self.a_s[2], self.alpha_s[2])
        A_34 = self.one_link(thetas[3] - math.pi/2, self.d_s[3], self.a_s[3], self.alpha_s[3])
        T_04 = A_01 * A_12 * self.rot_y * A_23 * A_34 * self.rot_y

        A_45 = self.one_link(thetas[4], self.d_s[4], self.a_s[4], self.alpha_s[4])
        A_56 = self.one_link(thetas[5] - math.pi/2, self.d_s[5], self.a_s[5], self.alpha_s[5])
        T_06 = T_04 * A_45 * A_56 * self.rot_x
        A_67 = self.one_link(thetas[6], self.d_s[6], self.a_s[6], self.alpha_s[6])
        T_07 = T_06 * A_67

        return T_07

    def one_link(self, theta, d, a, alpha):
        trans_x = np.matrix(np.identity(4))
        trans_x[0, 3] = a
        trans_z = np.matrix(np.identity(4))
        trans_z[2, 3] = d

        rot_z = np.matrix(np.array([[math.cos(theta), -1 * math.sin(theta), 0.0, 0.0],
                                    [math.sin(theta), math.cos(theta), 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]))

        rot_x = np.matrix(np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, math.cos(alpha), -1 * math.sin(alpha), 0.0],
                                    [0.0, math.sin(alpha), math.cos(alpha), 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]))

        trans_all = trans_z * rot_z * trans_x * rot_x
        return trans_all

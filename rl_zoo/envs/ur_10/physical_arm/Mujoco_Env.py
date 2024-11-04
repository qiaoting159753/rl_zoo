import mujoco_py
import math
import mujoco
from arm.forward_kinematics import Forward_Kinematics
import numpy as np
import time


def str_mj_arr(arr):
    return ' '.join(['%0.3f' % arr[i] for i in range(len(arr))])


class Arm_Mujoco:
    def __init__(self):
        xml_location = "/Users/tonyq/Desktop/arm.xml"
        model = mujoco_py.load_model_from_path(xml_location)
        self.sim = mujoco_py.MjSim(model)
        self.viewer = mujoco_py.MjViewer(self.sim)

        self.kine = Forward_Kinematics()

        # while 1:
        for i in range(5000):
            rand_position = np.random.uniform(low=-3.14, high=3.14, size=(7,))
            # rand_position = np.zeros(shape=(7,))
            # rand_position[4] = 0.5
            # rand_position[5] = -math.pi/498

            self.sim.data.qpos[0] = rand_position[0]
            self.sim.data.qpos[1] = rand_position[1]
            self.sim.data.qpos[2] = rand_position[2]
            self.sim.data.qpos[3] = -1 * rand_position[3]
            self.sim.data.qpos[4] = rand_position[4]
            self.sim.data.qpos[5] = rand_position[5]
            self.sim.data.qpos[6] = rand_position[6]

            self.sim.step()

            contacts = self.sim.data.contact
            contacted = False
            for coni in range(len(contacts)):
                con = contacts[coni]
                if con.dist != 0.0:
                    contacted = True
                    break

            if not contacted:
                trans = self.kine.forward_kinematics(rand_position)
                fx = (trans[0, 3]) / 100
                fy = (trans[1, 3]) / 100
                fz = (trans[2, 3]) / 100

                site = self.sim.data.site_xpos[0]
                sim_x = site[0] - 1.0
                sim_y = site[1]
                sim_z = site[2]

                # print("------------------------------------")
                # print(fx - sim_x)
                # print(fy - sim_y)
                # print(fz - sim_z)

                if (fx - sim_x) > 0.01:
                    print("X_Faulty")
                if (fy - sim_y) > 0.01:
                    print("Y_Faulty")
                if (fz - sim_z) > 0.01:
                    print("Z_Faulty")


            # self.viewer.render()

    def reset(self):
        print("reset")
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.sim.set_state()

    def step(self, actions, n_frame=5):
        print("Step")
        self.sim.data.ctrl[:] = actions
        for _ in range(n_frame):
            self.sim.step()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]





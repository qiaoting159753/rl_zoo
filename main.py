from loops import MFRL_Trainer
from envs import DMCSEnvironment
from utils import set_seed




def main():
    """
    SAC
    TQC
    Fully_Expand
    Hyper_SAC_Critic
    Hyper_TQC_Critic
    Hyper_SAC_all
    Hyper_TQC_all
    Hyper_SAC_actor
    Hyper_TQC_actor

    """
    # Random Seed
    seed = 10
    set_seed(seed)

    env = DMCSEnvironment("cheetah", "run")
    random_goal = False  # For real robot.
    device = "cpu"  # For mac training.

    agent_name = "SAC"

    action_dim = env.action_num
    state_dim = env.observation_space
    G = 5
    batch_size = 256

    trainer = MFRL_Trainer(env,
                           agent_name,
                           action_dim,
                           state_dim,
                           random_goal,
                           device,
                           G,
                           batch_size
                           )
    trainer.train()




# if __name__ == "__main__":
#     main()
#
# import torch
# import numpy as np
# # import torch.nn.functional as F
# # import math
# # from scipy.spatial.transform import Rotation
# # from arm.utils import matrix_to_euler_angles, matrix_to_quaternion
# # from envs.DMCS import DMCSEnvironment
#
# from envs.UR10_Kinematic_Env import UR10_Kinematic_Env
# from train_loop import Trainer

# if __name__ == "__main__":
#     env = UR10_Kinematic_Env()
    # trainer = Trainer(env, action_dim=6)
    # trainer.train()

# lists = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
# lists2 = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
# for conb1 in lists:
#     counter = 0
#     for i in range(1000):
#         euler_angles = 2 * (np.random.rand(3) - 0.5) * math.pi
#
#         rot = Rotation.from_euler(conb1, euler_angles)
#         matrix = np.expand_dims(rot.as_matrix(), axis=0)
#         matrix = torch.FloatTensor(matrix)
#
#         tensor_quat = matrix_to_quaternion(matrix)
#         tensor_quat = tensor_quat.squeeze().numpy()
#
#         distance = np.linalg.norm(np.roll(rot.as_quat(canonical=True), shift=1) - tensor_quat)
#
#         if distance > 0.01:
#             print("---------------")
#             print(np.roll(rot.as_quat(), shift=1))
#             print(tensor_quat)
#
#         # Successful
#         if distance < 0.01:
#             counter += 1
#
#     if counter < 10:
#         print("---------------------")
#         print(conb1)
#
#     print(counter)

# ############    Verification of the Query and Step    ####################
# evaluation = [[], [], []]
#
# for i in range(10):
#     state = env.reset()
#     for j in range(100):
#         # Parital
#         action = env.sample_action(1, 6)
#         action = action[0]
#         # Full for step.
#         next_state, reward, dones, _, info = env.step(action)
#         evaluation[0].append(info["quat_dist"])
#         evaluation[1].append(info["euler_dist"])
#         evaluation[2].append(info["euler_dist"] - info["quat_dist"])
#
#         # Partial state. Parital Action
#         state = np.expand_dims(state, axis=0)
#         action = np.expand_dims(action, axis=0)
#         state_tensor = torch.FloatTensor(state)
#         action_tensor = torch.FloatTensor(action)
#         pred_next, rd, dns = env.tensor_query(state_tensor, action_tensor)
#         pred_next = pred_next.numpy()
#         rd = rd.numpy()
#         dns = dns.numpy()
#
#         if dns[0] != dones:
#             print("lllllll")
#         # if distance > 0.001:
#         #     print(distance)
#         state = next_state
#
# eval_array = np.array(evaluation)
# # Save the metrics
# file_name = "data/Kinematic"
# np.savetxt(file_name + "_eval_rewards.csv", eval_array, delimiter=",")

# from envs.mujoco_env import Arm_Mujoco

# kine = Forward_Kinematics()
# theta = np.array(        [169.92, 121.48, 299.97, 211.52, 144.73, 34.86, 176.07])
# for_kine = (theta-degree_offset) * (math.pi / 180)
# pos, orient = kine.forward_kinematics(for_kine)

# env = Arm_Mujoco()

# ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
# while True:
#     data = ser.readline()# .decode().strip()
#     if data:
#         print(data)


# from Arm import Arm
# from Arm_Env import Arm_Env
# import dynamixel_sdk as dxl

# env = Arm_Env()
# env.sensor.get_position()

# env.arm.disable_all()

# for i in range(10):
#     state = env.reset()
#     for k in range(10):
#         action = env.sample_action()
#         env.step(action)

# arm.disable_all()
# arm._enable_all()

# fake_joints = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]])
# fake_joints = torch.FloatTensor(fake_joints)
# forward(fake_joints)

# port_handler = dxl.PortHandler("/dev/ttyUSB0")
# port_handler.openPort()
# port_handler.setBaudRate(1000000)
# packet_handler = dxl.PacketHandler(protocol_version=1.0)

# servo 0 position range: 400  - 800, home: 520
# servo 1 position range: 3535 - 164, home: 2024
# servo 2 position range: 20   - 990, home: 520
# servo 3 position range: 120  - 880, home: 510
# servo 4 position range: 0    - 1000, home: 520
# servo 5 position range: 160  - 860, home: 520
# servo 6 position range: 170  - 840, home: 520

# lists = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
# lists2 = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZYX', 'ZXY']
#
# for conb2 in lists2:
#     for conb1 in lists:
#         counter = 0
#         for i in range(1000):
#             euler_angles = 2 * (np.random.rand(3) - 0.5) * math.pi
#             rot = Rotation.from_euler(conb1, euler_angles)
#
#             matrix = np.expand_dims(rot.as_matrix(), axis=0)
#             matrix = torch.FloatTensor(matrix)
#             tensor_euler = matrix_to_euler_angles(matrix, conb2)
#             tensor_euler = tensor_euler.squeeze().numpy()
#             distance = np.linalg.norm(rot.as_euler(conb1) - tensor_euler)
#
#             if distance > 0.03:
#                 counter += 1
#
#         if counter < 10:
#             print("---------------------")
#             print(conb1)
#             print(conb2)
#             print(counter)

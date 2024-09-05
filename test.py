from scipy.spatial.transform import Rotation
import numpy as np
import math

eval_array = [[], [], []]

for i in range(1000):
    print("----------------------")
    a = (np.random.rand(3) - 0.5) * 2 * math.pi
    b = (np.random.rand(3) - 0.5) * 2 * math.pi

    # (x, y, z, w) -> (w, x, y, z) <a, b, c, d>
    rot_a = Rotation.from_euler(seq='xyz', angles=a)
    mat_a = rot_a.as_matrix()
    quat_a = np.roll(rot_a.as_quat(canonical=True), shift=1)
    # <e, f, g, h>
    rot_b = Rotation.from_euler(seq='xyz', angles=b)
    mat_b = rot_b.as_matrix()
    quat_b = np.roll(rot_b.as_quat(canonical=True), shift=1)

    # b = x * a
    # b * at = x
    x = np.matmul(mat_b, mat_a.transpose())
    rot_delta = Rotation.from_matrix(x)
    delta_euler = rot_delta.as_euler('xyz')
    eval_array[0].append(np.linalg.norm(delta_euler))

    a = quat_a[0]
    b = quat_a[1]
    c = quat_a[2]
    d = quat_a[3]

    e = quat_b[0]
    f = quat_b[1]
    g = quat_b[2]
    h = quat_b[3]

    q2m = np.matrix([[e, -1 * f, -1 * g, -1 * h],
                     [f, e, -1 * h, g],
                     [g, h, e, -1 * f],
                     [h, -1 * g, f, e]])
    q1i = np.matrix([[a], [-1 * b], [-1 * c], [-1 * d]])

    quat_dis = np.matmul(q2m, q1i)
    diff_rot = Rotation.from_quat(np.roll(np.squeeze(np.asarray(quat_dis)), shift=3))
    diff_euler = diff_rot.as_euler('xyz')
    eval_array[1].append(np.linalg.norm(diff_euler))

    # diff_euler = abs(diff_euler)
    # diff_euler_2 = math.pi * 2 - diff_euler
    # min_diff_euler_2 = np.minimum(diff_euler, diff_euler_2)
    # print(min_diff_euler_2)
    # quaternion_dist = math.sqrt(diff_euler[0] ** 2 + diff_euler[1] ** 2 + diff_euler[2] ** 2)

    p = np.array([quat_a[0], quat_a[1], quat_a[2], quat_a[3]])
    c_q = np.array([quat_b[0], -1 * quat_b[1], -1 * quat_b[2], -1 * quat_b[3]])
    z_0 = abs(p[0] * c_q[0] - p[1] * c_q[1] - p[2] * c_q[2] - p[3] * c_q[3])
    quaternion_dist = (2 * math.acos(z_0))
    print(quaternion_dist)
    eval_array[2].append(quaternion_dist)
    # print(quaternion_dist)

eval_array = np.array(eval_array)
np.savetxt("eval_rewards.csv", eval_array, delimiter=",")



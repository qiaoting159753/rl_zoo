import csv
import numpy as np

# load the imu data
file = open('data/foo.csv', 'r')
csvFile = csv.reader(file)
data = []
for lines in csvFile:
    data.append(lines)
data = np.array(data, dtype=float)

# [gyro_x, gyro_y, gyro_z, time_gyro, angle_x, angle_y, angle_z, time_angle, x, y, z, time_posi]

# Get rid of bias
static = data[:50]
means = np.mean(static, axis=0)
data[:, 0:3] -= means[0:3]
data[:, 4:7] -= means[4:7]
data[:, 8:11] -= means[8:11]


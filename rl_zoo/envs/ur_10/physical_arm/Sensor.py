import serial
import time
import numpy as np


class Sensor:
    """
    Sensor

    """

    def __init__(self):
        # Device
        baud_rate = 115200
        device_name = '/dev/tty.usbmodem21301'
        timeout = 1.0
        self.ser = serial.Serial(device_name, baud_rate, timeout=timeout)
        # ToDo: Check connection.
        self.create_time = time.perf_counter()

        # Monitor
        self.record_gyro_x = []
        self.record_gyro_y = []
        self.record_gyro_z = []
        self.time_gyro = []
        self.record_angle_x = []
        self.record_angle_y = []
        self.record_angle_z = []
        self.time_angle = []
        self.record_x = []
        self.record_y = []
        self.record_z = []
        self.time_posi = []

        # self.monitor = plt.figure()
        # self.length = 20
        # axis = plt.axes(xlim=(0, 4), ylim=(-2, 2))
        # self.line, = axis.plot([], [], lw=3)
        # self.line.set_data([], [])

        # Compute Current Pose
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_angle_x = 0.0
        self.curr_angle_y = 0.0
        self.curr_angle_z = 0.0

        # Helper variables.
        self.angle_time_start = 0.0
        self.acc_time_start = 0.0
        self.gyro_time_start = 0.0

    def save(self):
        """
        Save saved data.
        """
        all_data = [self.record_gyro_x, self.record_gyro_y, self.record_gyro_z, self.time_gyro,
                    self.record_angle_x, self.record_angle_y, self.record_angle_z, self.time_angle,
                    self.record_x, self.record_y, self.record_z, self.time_posi]
        all_data = np.array(all_data)
        all_data = np.transpose(all_data)
        np.savetxt("../data/foo.csv", all_data, delimiter=",")
        print("Data Saved!")

    def get_position(self):
        """
        Get Position.
        """
        while True:
            # Reading
            acc_updated = False
            gyro_updated = False
            angle_updated = False
            while True:
                a = self.ser.readline().decode()
                split_string = a.split("\t")

                if split_string[0] == 'Acc':
                    if len(split_string) == 4:
                        acc_x = float(split_string[1])
                        acc_y = float(split_string[2])
                        acc_z = float(split_string[3].split("\r")[0])
                        self.acc_time_start = time.perf_counter()
                        acc_updated = True
                if split_string[0] == "Gyro":
                    if len(split_string) == 4:
                        gyro_x = float(split_string[1])
                        gyro_y = float(split_string[2])
                        gyro_z = float(split_string[3].split("\r")[0])
                        self.gyro_time_start = time.perf_counter()
                        gyro_updated = True
                if split_string[0] == "Angle":
                    if len(split_string) == 4:
                        angle_x = float(split_string[1])
                        angle_y = float(split_string[2])
                        angle_z = float(split_string[3].split("\r")[0])
                        self.angle_time_start = time.perf_counter()
                        angle_updated = True

                if acc_updated and gyro_updated and angle_updated:
                    break

            # Save for calibration
            if (len(self.record_x)) == 50:
                print("Start !")

            # Append and save
            self.record_x.append(acc_x)
            self.record_y.append(acc_y)
            self.record_z.append(acc_z)
            self.record_gyro_x.append(gyro_x)
            self.record_gyro_y.append(gyro_y)
            self.record_gyro_z.append(gyro_z)
            self.record_angle_x.append(angle_x)
            self.record_angle_y.append(angle_y)
            self.record_angle_z.append(angle_z)
            self.time_posi.append(self.acc_time_start - self.create_time)
            self.time_gyro.append(self.gyro_time_start - self.create_time)
            self.time_angle.append(self.angle_time_start - self.create_time)

            if len(self.time_angle) % 100 == 0:
                self.save()
                print("Saved!")


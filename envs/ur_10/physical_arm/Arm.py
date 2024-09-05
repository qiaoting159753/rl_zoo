# from Servo import Servo
import dynamixel_sdk as dxl
from arm.address import adds

adds = adds['RX-28']
import numpy as np
import time

DXL_MOVING_STATUS_THRESHOLD = 10


class Arm:
    def __init__(self, num_motors=7, torque_scale=128, speed_scale = 128):
        # Full torque: 1023

        self.num_motors = num_motors
        self.port_handler = dxl.PortHandler("/dev/tty.usbserial-FT7WBBKL")
        self.port_handler.openPort()
        self.port_handler.setBaudRate(1000000)
        self.packet_handler = dxl.PacketHandler(protocol_version=1.0)

        # servo 0 position range: 400  - 800, home: 512
        # servo 1 position range: 3535 - 164, home: 2048
        # servo 2 position range: 20   - 990, home: 512
        # servo 3 position range: 120  - 880, home: 512
        # servo 4 position range: 0    - 1000, home: 512
        # servo 5 position range: 160  - 860, home: 512
        # servo 6 position range: 170  - 840, home: 512

        self.servo_ids = [0, 1, 2, 3, 4, 5, 6]
        # [400, 164, 20, 120, 0, 160, 170]
        self.position_mins = [520, 164, 20, 120, 0, 160, 170]
        self.position_maxs = [800, 3535, 990, 880, 1000, 860, 840]
        self.position_home = [520, 2024, 520, 510, 520, 520, 512]
        self.scales = [400, 3371, 970, 760, 1000, 700, 670]

        print("Initialize the arm, ping all: ")
        success = self.ping_all()

        data = np.ones((self.num_motors,)) * torque_scale
        self._bulk_write(name="torque_limit", length=2, data=data)

        data = np.ones((self.num_motors,)) * speed_scale
        self._bulk_write(name="moving_speed", length=2, data=data)

        if success:
            print("First Communication success, enable all")
            self.enable_all()

    def ping_all(self):
        for i_d in self.servo_ids:
            model_number, res, err = self.packet_handler.ping(self.port_handler, i_d)
            if not (res == dxl.COMM_SUCCESS):
                return False
        return True

    def disable_all(self):
        switches = np.zeros((self.num_motors,))
        self._bulk_write(name="torque_enable", length=1, data=switches)

    def enable_all(self):
        switches = np.ones((self.num_motors,))
        self._bulk_write(name="torque_enable", length=1, data=switches)

    def go_home(self, interpolate=100):
        curr_position = self.get_positions()
        curr_position = np.array(curr_position)

        target_posiiton = np.array(self.position_home)
        # target_posiiton = curr_position + 50

        big_action = target_posiiton - curr_position
        sub_action = big_action / interpolate
        # make a trajectory.
        traj = []
        for i in range(interpolate):
            intrim_step = curr_position + sub_action
            traj.append(intrim_step)
            curr_position = intrim_step

        # Execute the trajectory.
        for j in range(interpolate):
            self.bulk_move(traj[j])

    def bulk_move(self, targets, wait=True, timeout=2):
        # Wait for a while
        if wait:
            arm_moving = True
            start_time = time.time()
            while arm_moving:
                arm_moving = self.is_moving()
                elapsed = time.time() - start_time
                if (elapsed > timeout) or (not arm_moving):
                    break
                time.sleep(1)
        all_good = self._verify_steps(targets)
        if not all_good:
            print(targets)
            print("Give up due to Out of range")
            return
        # Move the joints.
        # for i_d in self.servo_ids:
        curr_position = self.get_positions()
        for i in range(len(curr_position)):
            if abs(curr_position[i]-targets[i]) > 100:
                print("Give up due to large movement in a short period.")
                return
        self._bulk_write(name="goal_position", length=2, data=targets)

    def _verify_steps(self, target_steps):
        counter = 0
        all_good = True
        for i_d in self.servo_ids:
            step = target_steps[counter]
            verify_step = (self.position_mins[i_d] <= step <= self.position_maxs[i_d])
            if not verify_step:
                print("Exceed step range at " + str(counter))
                all_good = False
            counter = counter + 1
        return all_good

    def get_loads(self):
        all_res = self._bulk_read_v2(name="current_load", length=2)
        return all_res

    def get_velocities(self):
        all_res = self._bulk_read_v2(name="current_velocity", length=2)
        return all_res

    def get_positions(self):
        all_res = self._bulk_read(name="current_position", length=2)
        return all_res

    def is_moving(self):
        for i_d in self.servo_ids:
            current_position, res, err = self.packet_handler.read1ByteTxRx(self.port_handler, i_d,
                                                                           adds['current_position'])
            goal_position, res, err = self.packet_handler.read1ByteTxRx(self.port_handler, i_d, adds['goal_position'])
            is_moving = (abs(current_position - goal_position) > DXL_MOVING_STATUS_THRESHOLD)
            if is_moving:
                return True
        return False

    def _bulk_read(self, name, length):
        readings = []
        for i_d in self.servo_ids:
            data_read = -1
            if length == 1:
                data_read, res, err = self.packet_handler.read1ByteTxRx(self.port_handler, i_d, adds[name])
            elif length == 2:
                data_read, res, err = self.packet_handler.read2ByteTxRx(self.port_handler, i_d, adds[name])
            if res != 0:
                print("Error in bulk_read")
            readings.append(data_read)
        return readings

    def _bulk_write(self, name, length, data):
        counter = 0
        for i_d in self.servo_ids:
            if length == 1:
                res, err = self.packet_handler.write1ByteTxRx(self.port_handler, i_d, adds[name], int(data[counter]))
            elif length == 2:
                res, err = self.packet_handler.write2ByteTxRx(self.port_handler, i_d, adds[name], int(data[counter]))
            counter += 1
            if res:
                print("bulk write error code: ")
                print(err)

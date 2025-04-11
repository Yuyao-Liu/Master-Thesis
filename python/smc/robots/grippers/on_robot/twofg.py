from smc.robots.grippers.abstract_gripper import AbstractGripper
import time
import xmlrpc.client


class OnRobotDevice:
    """
    Generic OnRobot device object
    """

    cb = None

    def __init__(self, Global_cbip="192.168.1.1"):
        # try to get Computebox IP address
        try:
            self.Global_cbip = Global_cbip
        except NameError:
            print("Global_cbip is not defined!")

    def getCB(self):
        try:
            self.cb = xmlrpc.client.ServerProxy(
                "http://" + str(self.Global_cbip) + ":41414/"
            )
            return self.cb
        except TimeoutError:
            print("Connection to ComputeBox failed!")


"""
XML-RPC library for controlling OnRobot devcies from Doosan robots

Global_cbip holds the IP address of the compute box, needs to be defined by the end user
"""


class TwoFG(AbstractGripper):
    """
    This class is for handling the 2FG device
    """

    cb = None

    def __init__(self):
        dev = OnRobotDevice()
        self.cb = dev.getCB()
        # Device ID
        self.TWOFG_ID = 0xC0

        # Connection
        self.CONN_ERR = -2  # Connection failure
        self.RET_OK = 0  # Okay
        self.RET_FAIL = -1  # Error

        # arguments moved to these parameters because you almost certainly won't need this
        self.t_index = 0
        self.wait_for_grip = False
        self.speed = 10  # [m/s]
        # self.gripping_force = 20 # [N]
        self.gripping_force = 140  # [N]
        self.max_width = self.get_max_exposition()
        self.min_width = self.get_min_exposition()

    def isConnected(self):
        """
        Returns with True if 2FG device is connected, False otherwise

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @return: True if connected, False otherwise
        @rtype: bool
        """
        try:
            IsTwoFG = self.cb.cb_is_device_connected(self.t_index, self.TWOFG_ID)
        except TimeoutError:
            IsTwoFG = False

        if IsTwoFG is False:
            print("No 2FG device connected on the given instance")
            return False
        else:
            return True

    def isBusy(self):
        """
        Gets if the gripper is busy or not

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @type t_index: int

        @rtype: bool
        @return: True if busy, False otherwise
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        return self.cb.twofg_get_busy(self.t_index)

    def setWaitForGrip(self, wait_for_grip: bool):
        self.wait_for_grip = wait_for_grip

    def open(self):
        self.move(self.max_width)

    def close(self):
        self.move(self.min_width)

    def setSpeed(speed: float):
        self.speed = speed

    def setGrippingForce(gripping_force: float):
        self.gripping_force = gripping_force

    def isGripped(self):
        """
        Gets if the gripper is gripping or not

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @type t_index: int

        @rtype: bool
        @return: True if gripped, False otherwise
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        return self.cb.twofg_get_grip_detected()

    def getStatus(self):
        """
        Gets the status of the gripper

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @type t_index: int

        @rtype: int
        @return: Status code of the device
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        status = self.cb.twofg_get_status(self.t_index)
        return status

    def get_exposition(self):
        """
        Returns with current external width

        @param MOVE TO PROPERTY: t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @return: External width in mm
        @rtype: float
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        extWidth = self.cb.twofg_get_external_width(self.t_index)
        return extWidth

    def get_min_exposition(self):
        """
        Returns with current minimum external width

        @param  MOVE TO PROPERTY: t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @return: Minimum external width in mm
        @rtype: float
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        extMinWidth = self.cb.twofg_get_min_external_width(self.t_index)
        return extMinWidth

    def get_max_exposition(self):
        """
        Returns with current maximum external width

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @return: Maximum external width in mm
        @rtype: float
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        extMaxWidth = self.cb.twofg_get_max_external_width(self.t_index)
        return extMaxWidth

    def get_force(self):
        """
        Returns with current force

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @return: Force in N
        @rtype: float
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        currForce = self.cb.twofg_get_force(self.t_index)
        return currForce

    def stop(self):
        """
        Stop the grippers movement

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @type t_index: int
        """
        if self.isConnected() is False:
            return self.CONN_ERR
        self.cb.twofg_stop(self.t_index)

    def move(self, position, speed=None, gripping_force=None):
        """
        Makes an external grip with the gripper to the desired position

        @param MOVED TO PROPERTY t_index: The position of the device (0 for single, 1 for dual primary, 2 for dual secondary)
        @param position: The width to move the gripper to in mm's
        @type position: float
        @param gripping_force: The force to move the gripper width in N
        @type gripping_force: float
        @param speed: The speed of the gripper in %
        @type speed: int
        @type self.wait_for_grip: bool
        @param self.wait_for_grip: wait for the grip to end or not?
        """

        if self.isConnected() is False:
            return self.CONN_ERR

        if speed == None:
            speed = self.speed

        if gripping_force == None:
            gripping_force = self.gripping_force

        # Sanity check
        # self.max_width = self.get_max_exposition()
        # self.min_width = self.get_min_exposition()
        if position > self.max_width or position < self.min_width:
            print(
                "Invalid 2FG width parameter, "
                + str(self.max_width)
                + " - "
                + str(self.min_width)
                + " is valid only"
            )
            return self.RET_FAIL

        if gripping_force > 140 or gripping_force < 20:
            print("Invalid 2FG force parameter, 20-140 is valid only")
            return self.RET_FAIL

        if speed > 100 or speed < 10:
            print("Invalid 2FG speed parameter, 10-100 is valid only")
            return self.RET_FAIL

        self.cb.twofg_grip_external(
            self.t_index, float(position), int(gripping_force), int(speed)
        )

        if self.wait_for_grip:
            tim_cnt = 0
            fbusy = self.isBusy()
            while fbusy:
                time.sleep(0.1)
                fbusy = self.isBusy()
                tim_cnt += 1
                if tim_cnt > 30:
                    print("2FG external grip command timeout")
                    break
            else:
                # Grip detection
                grip_tim = 0
                gripped = self.isGripped()
                while not gripped:
                    time.sleep(0.1)
                    gripped = self.isGripped()
                    grip_tim += 1
                    if grip_tim > 20:
                        print("2FG external grip detection timeout")
                        break
                else:
                    return self.RET_OK
                return self.RET_FAIL
            return self.RET_FAIL
        else:
            return self.RET_OK


if __name__ == "__main__":
    device = OnRobotDevice()
    device.getCB()
    gripper_2FG7 = TwoFG(device)
    # gripper_2FG7.grip(0, position=37.0)
    # gripper_2FG7.grip(0, position=37.0)
    gripper_2FG7.move(10.0)
    time.sleep(1.0)
    gripper_2FG7.close()
    time.sleep(1.0)
    gripper_2FG7.open()
    print("Connection check: ", gripper_2FG7.isConnected())

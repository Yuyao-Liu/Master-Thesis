from ur_simple_control.robots.robotmanager_abstract import RobotManagerAbstract
from ur_simple_control.robots.single_arm_interface import SingleArmInterface
from ur_simple_control.util.get_model import get_model
import numpy as np
import pinocchio as pin
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface


# NOTE: this one assumes a jaw gripper
class RobotManagerUR5e(SingleArmInterface):
    def __init__(self, args):
        self.args = args
        self.model, self.collision_model, self.visual_model, self.data = get_model()
        self._MAX_ACCELERATION = 1.7  # const
        self._MAX_QD = 3.14  # const
        # NOTE: this is evil and everything only works if it's set to 1
        # you really should control the acceleration via the acceleration argument.
        # we need to set it to 1.0 with ur_rtde so that's why it's here and explicitely named
        self._speed_slider = 1.0  # const

        self.connectToRobot()

        self.setInitialPose()
        if not self.args.real and self.args.start_from_current_pose:
            self.rtde_receive = RTDEReceiveInterface(args.robot_ip)
            q = self.rtde_receive.getActualQ()
            q = np.array(q)
            self.q = q
            if args.visualize_manipulator:
                self.visualizer_manager.sendCommand({"q": q})

    def setInitialPose(self):
        if not self.args.real and self.args.start_from_current_pose:
            self.rtde_receive = RTDEReceiveInterface(args.robot_ip)
            self.q = np.array(self.rtde_receive.getActualQ())
            if self.args.visualize_manipulator:
                self.visualizer_manager.sendCommand({"q": self.q})
        if not self.args.real and not self.args.start_from_current_pose:
            self.q = pin.randomConfiguration(
                self.model, -1 * np.ones(self.model.nq), np.ones(self.model.nq)
            )
        if self.args.real:
            self.q = np.array(self.rtde_receive.getActualQ())

    def connectToRobot(self):
        if self.args.real:
            # NOTE: you can't connect twice, so you can't have more than one RobotManager per robot.
            # if this produces errors like "already in use", and it's not already in use,
            # try just running your new program again. it could be that the socket wasn't given
            # back to the os even though you've shut off the previous program.
            print("CONNECTING TO UR5e!")
            self.rtde_control = RTDEControlInterface(self.args.robot_ip)
            self.rtde_receive = RTDEReceiveInterface(self.args.robot_ip)
            self.rtde_io = RTDEIOInterface(self.args.robot_ip)
            self.rtde_io.setSpeedSlider(self.args.speed_slider)
            # NOTE: the force/torque sensor just has large offsets for no reason,
            # and you need to minus them to have usable readings.
            # we provide this with calibrateFT
            self.wrench_offset = self.calibrateFT()
        else:
            self.wrench_offset = np.zeros(6)

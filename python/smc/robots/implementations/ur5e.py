from smc.robots.interfaces.force_torque_sensor_interface import (
    ForceTorqueOnSingleArmWrist,
)
from smc.robots.grippers.robotiq.robotiq_gripper import RobotiqGripper
from smc.robots.grippers.rs485_robotiq.rs485_robotiq import RobotiqHand
from smc.robots.grippers.on_robot.twofg import TwoFG
from smc.robots.abstract_robotmanager import AbstractRealRobotManager
from smc.robots.abstract_simulated_robotmanager import AbstractSimulatedRobotManager

import numpy as np
import pinocchio as pin
from importlib.resources import files
from importlib.util import find_spec
import time
import argparse
from os import path

if find_spec("rtde_control"):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    from rtde_io import RTDEIOInterface


# NOTE: this one assumes a jaw gripper
class AbstractUR5eRobotManager(ForceTorqueOnSingleArmWrist):
    def __init__(self, args):
        if args.debug_prints:
            print("AbstractUR5eRobotManager init")
        self._model, self._collision_model, self._visual_model, self._data = get_model()
        self._ee_frame_id = self.model.getFrameId("tool0")
        self._MAX_ACCELERATION = 1.7  # const
        self._MAX_QD = 3.14  # const
        super().__init__(args)


class SimulatedUR5eRobotManager(
    AbstractSimulatedRobotManager, AbstractUR5eRobotManager
):
    def __init__(self, args):
        if args.debug_prints:
            print("SimulatedRobotManagerUR5e init")
        super().__init__(args)

    # NOTE: overriding wrench stuff here
    # there can be a debated whether there should be a simulated forcetorquesensorinterface,
    # but it's annoying as hell and there is no immediate benefit in solving this problem
    def _updateWrench(self):
        self._wrench_base = np.random.random(6)
        # NOTE: create a robot_math module, make this mapping a function called
        # mapse3ToDifferent frame or something like that
        mapping = np.zeros((6, 6))
        mapping[0:3, 0:3] = self._T_w_e.rotation
        mapping[3:6, 3:6] = self._T_w_e.rotation
        self._wrench = mapping.T @ self._wrench_base

    def zeroFtSensor(self) -> None:
        self._wrench_bias = np.zeros(6)

    def setInitialPose(self):
        if self.args.start_from_current_pose:
            rtde_receive = RTDEReceiveInterface(self.args.robot_ip)
            self._q = np.array(rtde_receive.getActualQ())
        else:
            self._q = pin.randomConfiguration(
                self.model, self.model.lowerPositionLimit, self.model.upperPositionLimit
            )


class RealUR5eRobotManager(AbstractUR5eRobotManager, AbstractRealRobotManager):
    def __init__(self, args: argparse.Namespace):
        if args.debug_prints:
            print("RealUR5eRobotManager init")
        # NOTE: UR's sleep slider is evil and nothing works unless if it's set to 1.0!!!
        # you have to control the acceleration via the acceleration argument.
        # we need to set it to 1.0 with ur_rtde so that's why it's here and explicitely named
        self._speed_slider = 1.0  # const
        self._rtde_control: RTDEControlInterface
        self._rtde_receive: RTDEReceiveInterface
        self._rtde_io: RTDEIOInterface
        # TODO: copy-pasted from ForceTorqueSensorInterface's __init__
        # this should be inited automatically, it's not, todo is to make it work
        self._wrench_base: np.ndarray = np.zeros(6)
        self._wrench: np.ndarray = np.zeros(6)
        # NOTE: wrench bias will be defined in the frame your sensor's gives readings
        self._wrench_bias: np.ndarray = np.zeros(6)
        self._T_w_e = pin.SE3.Identity()
        super().__init__(args)

    def connectToGripper(self):
        if (self.args.gripper == "none") or not self.args.real:
            self.gripper = None
            return
        if self.args.gripper == "robotiq":
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.args.robot_ip, 63352)
            self.gripper.activate()
        if self.args.gripper == "onrobot":
            self.gripper = TwoFG()
        if self.args.gripper == "rs485":
            self.gripper = RobotiqHand()
            self.gripper.connect(self.args.robot_ip, 54321)
            self.gripper.reset()
            self.gripper.activate()
            result = self.gripper.wait_activate_complete()
            if result != 0x31:
                print("ERROR: can't activate gripper!! - exiting")
                self.gripper.disconnect()
                exit()

    def setInitialPose(self):
        self._q = np.array(self._rtde_receive.getActualQ())

    def connectToRobot(self):
        # NOTE: you can't connect twice, so you can't have more than one RobotManager per robot.
        # if this produces errors like "already in use", and it's not already in use,
        # try just running your new program again. it could be that the socket wasn't given
        # back to the os even though you've shut off the previous program.
        print("CONNECTING TO UR5e!")
        self._rtde_control = RTDEControlInterface(self.args.robot_ip)
        self._rtde_receive = RTDEReceiveInterface(self.args.robot_ip)
        self._rtde_io = RTDEIOInterface(self.args.robot_ip)
        self._rtde_io.setSpeedSlider(self._speed_slider)
        # NOTE: the force/torque sensor just has large offsets for no reason,
        # and you need to minus them to have usable readings.
        # we provide this with calibrateFT
        # self.wrench_offset = self.calibrateFT(self._dt)
        self.calibrateFT(self._dt)

    def setSpeedSlider(self, value):
        """
        setSpeedSlider
        ---------------
        update in all places
        NOTE: THIS IS EVIL AND NOTHING WORKS UNLESS IT'S SET TO 1.0!!!
                USE AT YOUR PERIL IF YOU DON'T KNOW WHAT IT DOES (i don't)
        """
        assert value <= 1.0 and value > 0.0
        if not self.args.pinocchio_only:
            self._rtde_io.setSpeedSlider(value)
        self.speed_slider = value

    def _updateQ(self):
        q = self._rtde_receive.getActualQ()
        self._q = np.array(q)

    def _updateV(self):
        v = self._rtde_receive.getActualQd()
        self._v = np.array(v)

    def _updateWrench(self):
        if not self.args.real:
            self._wrench_base = np.random.random(6)
        else:
            # NOTE: UR5e's ft-sensors gives readings in robot's base frame
            self._wrench_base = (
                np.array(self._rtde_receive.getActualTCPForce()) - self._wrench_bias
            )
        # NOTE: we define the default wrench to be given in the end-effector frame
        mapping = np.zeros((6, 6))
        mapping[0:3, 0:3] = self._T_w_e.rotation
        mapping[3:6, 3:6] = self._T_w_e.rotation
        self._wrench = mapping.T @ self._wrench_base

    def zeroFtSensor(self):
        self._rtde_control.zeroFtSensor()

    def sendVelocityCommandToReal(self, v):
        # speedj(qd, scalar_lead_axis_acc, hangup_time_on_command)
        self._rtde_control.speedJ(v, self._acceleration, self._dt)

    def stopRobot(self):
        self._rtde_control.speedStop(1)
        print("sending a stopj as well")
        self._rtde_control.stopJ(1)
        print("putting it to freedrive for good measure too")
        print("stopping via freedrive lel")
        self._rtde_control.freedriveMode()
        time.sleep(0.5)
        self._rtde_control.endFreedriveMode()

    def setFreedrive(self):
        self._rtde_control.freedriveMode()

    def unSetFreedrive(self):
        self._rtde_control.endFreedriveMode()


def get_model(
    with_gripper_joints=False,
) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel, pin.Data]:

    urdf_path_relative = files("smc.robots.robot_descriptions.urdf").joinpath(
        "ur5e_with_robotiq_hande_FIXED_PATHS.urdf"
    )
    urdf_path_absolute = path.abspath(urdf_path_relative)
    mesh_dir = files("smc")
    mesh_dir = mesh_dir.joinpath("robots")
    mesh_dir_absolute = path.abspath(mesh_dir)

    shoulder_trans = np.array([0, 0, 0.1625134425523304])
    shoulder_rpy = np.array([-0, 0, 5.315711138647629e-08])
    shoulder_se3 = pin.SE3(pin.rpy.rpyToMatrix(shoulder_rpy), shoulder_trans)

    upper_arm_trans = np.array([0.000300915150907851, 0, 0])
    upper_arm_rpy = np.array([1.571659987714477, 0, 1.155342090832558e-06])
    upper_arm_se3 = pin.SE3(pin.rpy.rpyToMatrix(upper_arm_rpy), upper_arm_trans)

    forearm_trans = np.array([-0.4249536100418752, 0, 0])
    forearm_rpy = np.array([3.140858652067472, 3.141065383898231, 3.141581851193229])
    forearm_se3 = pin.SE3(pin.rpy.rpyToMatrix(forearm_rpy), forearm_trans)

    wrist_1_trans = np.array(
        [-0.3922353894477613, -0.001171506236920081, 0.1337997346972175]
    )
    wrist_1_rpy = np.array(
        [0.008755445624588536, 0.0002860523431017214, 7.215921353974553e-06]
    )
    wrist_1_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_1_rpy), wrist_1_trans)

    wrist_2_trans = np.array(
        [5.620166987673597e-05, -0.09948910981796041, 0.0002201494606859632]
    )
    wrist_2_rpy = np.array([1.568583530823855, 0, -3.513049549874747e-07])
    wrist_2_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_2_rpy), wrist_2_trans)

    wrist_3_trans = np.array(
        [9.062061300900664e-06, 0.09947787349620175, 0.0001411778743239612]
    )
    wrist_3_rpy = np.array([1.572215514545703, 3.141592653589793, 3.141592633687631])
    wrist_3_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_3_rpy), wrist_3_trans)

    model = None
    collision_model = None
    visual_model = None
    # this command just calls the ones below it. both are kept here
    # in case pinocchio people decide to change their api.
    # model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path_absolute, mesh_dir_absolute)
    model = pin.buildModelFromUrdf(urdf_path_absolute)
    visual_model = pin.buildGeomFromUrdf(
        model, urdf_path_absolute, pin.GeometryType.VISUAL, None, mesh_dir_absolute
    )
    collision_model = pin.buildGeomFromUrdf(
        model, urdf_path_absolute, pin.GeometryType.COLLISION, None, mesh_dir_absolute
    )

    # for whatever reason the hand-e files don't have/
    # meshcat can't read scaling information.
    # so we scale manually,
    # and the stupid gripper is in milimeters
    for geom in visual_model.geometryObjects:
        if "hand" in geom.name:
            s = geom.meshScale
            # this looks exactly correct lmao
            s *= 0.001
            geom.meshScale = s
    for geom in collision_model.geometryObjects:
        if "hand" in geom.name:
            s = geom.meshScale
            # this looks exactly correct lmao
            s *= 0.001
            geom.meshScale = s

    # updating joint placements.
    model.jointPlacements[1] = shoulder_se3
    model.jointPlacements[2] = upper_arm_se3
    model.jointPlacements[3] = forearm_se3
    model.jointPlacements[4] = wrist_1_se3
    model.jointPlacements[5] = wrist_2_se3
    model.jointPlacements[6] = wrist_3_se3
    # TODO: fix where the fingers end up by setting a better position here (or maybe not here idk)
    if not with_gripper_joints:
        model = pin.buildReducedModel(model, [7, 8], np.zeros(model.nq))
    data = pin.Data(model)

    return model, collision_model, visual_model, data


def getGripperlessUR5e():
    import example_robot_data

    robot = example_robot_data.load("ur5")

    shoulder_trans = np.array([0, 0, 0.1625134425523304])
    shoulder_rpy = np.array([-0, 0, 5.315711138647629e-08])
    shoulder_se3 = pin.SE3(pin.rpy.rpyToMatrix(shoulder_rpy), shoulder_trans)

    upper_arm_trans = np.array([0.000300915150907851, 0, 0])
    upper_arm_rpy = np.array([1.571659987714477, 0, 1.155342090832558e-06])
    upper_arm_se3 = pin.SE3(pin.rpy.rpyToMatrix(upper_arm_rpy), upper_arm_trans)

    forearm_trans = np.array([-0.4249536100418752, 0, 0])
    forearm_rpy = np.array([3.140858652067472, 3.141065383898231, 3.141581851193229])
    forearm_se3 = pin.SE3(pin.rpy.rpyToMatrix(forearm_rpy), forearm_trans)

    wrist_1_trans = np.array(
        [-0.3922353894477613, -0.001171506236920081, 0.1337997346972175]
    )
    wrist_1_rpy = np.array(
        [0.008755445624588536, 0.0002860523431017214, 7.215921353974553e-06]
    )
    wrist_1_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_1_rpy), wrist_1_trans)

    wrist_2_trans = np.array(
        [5.620166987673597e-05, -0.09948910981796041, 0.0002201494606859632]
    )
    wrist_2_rpy = np.array([1.568583530823855, 0, -3.513049549874747e-07])
    wrist_2_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_2_rpy), wrist_2_trans)

    wrist_3_trans = np.array(
        [9.062061300900664e-06, 0.09947787349620175, 0.0001411778743239612]
    )
    wrist_3_rpy = np.array([1.572215514545703, 3.141592653589793, 3.141592633687631])
    wrist_3_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_3_rpy), wrist_3_trans)

    robot.model.jointPlacements[1] = shoulder_se3
    robot.model.jointPlacements[2] = upper_arm_se3
    robot.model.jointPlacements[3] = forearm_se3
    robot.model.jointPlacements[4] = wrist_1_se3
    robot.model.jointPlacements[5] = wrist_2_se3
    robot.model.jointPlacements[6] = wrist_3_se3
    data = pin.Data(robot.model)
    return robot.model, robot.collision_model, robot.visual_model, data

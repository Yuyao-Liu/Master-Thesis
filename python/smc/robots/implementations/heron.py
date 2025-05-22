from smc.robots.abstract_robotmanager import AbstractRealRobotManager
from smc.robots.abstract_simulated_robotmanager import AbstractSimulatedRobotManager
from smc.robots.interfaces.force_torque_sensor_interface import (
    ForceTorqueOnSingleArmWrist,
)
from smc.robots.interfaces.mobile_base_interface import (
    get_mobile_base_model,
)
from smc.robots.interfaces.whole_body_single_arm_interface import (
    SingleArmWholeBodyInterface,
)
from smc.robots.implementations.ur5e import get_model
from smc.robots.grippers.robotiq.robotiq_gripper import RobotiqGripper
from smc.robots.grippers.rs485_robotiq.rs485_robotiq import RobotiqHand

from argparse import Namespace
import numpy as np
import pinocchio as pin
import time

from importlib.util import find_spec

if find_spec("rtde_control"):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    from rtde_io import RTDEIOInterface


class AbstractHeronRobotManager(
    ForceTorqueOnSingleArmWrist, SingleArmWholeBodyInterface
):
    def __init__(self, args):
        if args.debug_prints:
            print("AbstractHeronRobotManager init")
        self._model, self._collision_model, self._visual_model, self._data = (
            heron_approximation()
        )
        self._ee_frame_id = self.model.getFrameId("tool0")
        self._base_frame_id = self.model.getFrameId("mobile_base")
        # TODO: CHANGE THIS TO REAL VALUES
        self._MAX_ACCELERATION = 1.7  # const
        self._MAX_QD = 3.14  # const
        self._comfy_configuration = np.array(
            [
                0.0,
                0.0,
                1.0,
                0.0,
                1.54027569e00,
                -1.95702042e00,
                1.46127540e00,
                -1.07315435e00,
                -1.61189968e00,
                -1.65158907e-03,
            ]
        )
        super().__init__(args)


class SimulatedHeronRobotManager(
    AbstractHeronRobotManager, AbstractSimulatedRobotManager
):
    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("SimulatedRobotManagerHeron init")
        super().__init__(args)

    # NOTE: overriding wrench stuff here
    # there can be a debated whether there should be a simulated forcetorquesensorinterface,
    # but it's annoying as hell and there is no immediate benefit in solving this problem
    def _updateWrench(self) -> None:
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
            # TODO: read position from localization topic,
            # put that into _q
            # HAS TO BE [x, y, cos(theta), sin(theta)] due to pinocchio's
            # representation of planar joint state
            self._q = np.zeros(self.nq)
            rtde_receive = RTDEReceiveInterface(self.args.robot_ip)
            self._q[4:] = np.array(rtde_receive.getActualQ())
        else:
            self._q = pin.randomConfiguration(
                self.model, self.model.lowerPositionLimit, self.model.upperPositionLimit
            )
            # pin.RandomConfiguration does not work well for planar joint,
            # or at least i remember something along those lines being the case
            self._q[0] = np.random.random()
            self._q[1] = np.random.random()
            theta = np.random.random() * 2 * np.pi - np.pi
            self._q[2] = np.cos(theta)
            self._q[3] = np.sin(theta)


class RealHeronRobotManager(AbstractHeronRobotManager, AbstractRealRobotManager):
    def __init__(self, args):
        if args.debug_prints:
            print("RealHeronRobotManager init")
        self._speed_slider = 1.0  # const
        self._rtde_control: RTDEControlInterface
        self._rtde_receive: RTDEReceiveInterface
        self._rtde_io: RTDEIOInterface
        self._wrench_base: np.ndarray = np.zeros(6)
        self._wrench: np.ndarray = np.zeros(6)
        # NOTE: wrench bias will be defined in the frame your sensor's gives readings
        self._wrench_bias: np.ndarray = np.zeros(6)
        self._T_w_e = pin.SE3.Identity()
        super().__init__(args)
        self._v_cmd = np.zeros(self.model.nv)
        # raise NotImplementedError
        # TODO: instantiate topics for reading base position /ekf_something
        # TODO: instantiate topics for sending base velocity commands /cmd_vel

    def connectToGripper(self):
        if (self.args.gripper == "none") or not self.args.real:
            self.gripper = None
            return
        if self.args.gripper == "robotiq":
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.args.robot_ip, 63352)
            self.gripper.activate()
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
        # TODO: read position from localization topic,
        # put that into _q
        # HAS TO BE [x, y, cos(theta), sin(theta)] due to pinocchio's
        # representation of planar joint state
        self._q = np.zeros(self.nq)
        self._q[4:] = np.array(self._rtde_receive.getActualQ())

    def connectToRobot(self):
        # NOTE: you can't connect twice, so you can't have more than one RobotManager per robot.
        # if this produces errors like "already in use", and it's not already in use,
        # try just running your new program again. it could be that the socket wasn't given
        # back to the os even though you've shut off the previous program.
        print("CONNECTING TO UR5e!")
        self._rtde_control = RTDEControlInterface(self.args.robot_ip)
        self._rtde_receive = RTDEReceiveInterface(self.args.robot_ip)
        self._rtde_io = RTDEIOInterface(self.args.robot_ip)
        # self._rtde_io.setSpeedSlider(self.args.speed_slider)
        # NOTE: the force/torque sensor just has large offsets for no reason,
        # and you need to minus them to have usable readings.
        # we provide this with calibrateFT
        self.calibrateFT(self._dt)
        # TODO:: instantiate topic for reading base position,
        # i.e. the localization topic
        # raise NotImplementedError

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

    # NOTE: handled by a topic callback
    def _updateQ(self):
        pass

    def _updateV(self):
        v = self._rtde_receive.getActualQd()
        # TODO: read base velocity from localization topic(?)
        # or the IMU. MAKE SURE that the reading (is mapped in)to
        # the correct frame - ex. on heron the imu is in the camera frame.
        # put that into _v
        # HAS TO BE [dot_x, dot_y, dot_theta] and in the first 3 values
        self._v[3:] = np.array(v)

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

    def sendVelocityCommand(self, v):
        # print(self.T_w_e)
        # print("send v_cmd")
        # speedj(qd, scalar_lead_axis_acc, hangup_time_on_command)
        assert type(v) == np.ndarray
        assert len(v) == self.model.nv
        # v_cmd_to_real = np.clip(v, -1 * self._max_v, self._max_v)
        non_zero_mask = self._max_v != 0
        K = max(np.abs(v[non_zero_mask]) / np.abs(self._max_v[non_zero_mask]))
        v = v / max(1.0, K)
        v_cmd_to_real = np.clip(v, -1 * self._max_v, self._max_v)
        self._v_cmd = v_cmd_to_real
        # TODO: send the velocity command to the base by publishing to the
        # /vel_cmd topic
        # NOTE: look at sendVelocityCommand in RobotManager,
        # understand relationship between sendVelocityCommand and sendVelocityCommandToReal

        # NOTE: that sendVelocityCommand is overridden
        # in SingleArmWholeBodyInterface
        # (which AbstractHeronRobotManager inherits from) due
        # to different control modes resulting in different sizes
        # of qs and vs.
        # TODO: make sure that this function and they either work together,
        # or override again if they can't work toherther
    def sendVelocityCommandToReal(self, v):
        self.sendVelocityCommand(v)
        
    def stopRobot(self):
        self._rtde_control.speedStop(1)
        print("sending a stopj as well")
        self._rtde_control.stopJ(1)
        print("putting it to freedrive for good measure too")
        print("stopping via freedrive lel")
        self._rtde_control.freedriveMode()
        time.sleep(0.5)
        self._rtde_control.endFreedriveMode()
        self._v_cmd[:] = 0.0
        # raise NotImplementedError
        # TODO: we need to stop be the base as well.
        # option 1) send zero velocity commands.
        # but then make sure that it doesn't keep going forward
        # with a near-zero velocity (this happens on UR5e, that's why
        # the freedrive is started because it actually stops the arm)
        # option 2) programaticaly activate the emergency button
        # option 3) stop just the arm, do nothing for the base, and
        # clearly document that the robot has to be stopped by manually
        # pressing the emergency button

    def setFreedrive(self):
        self._rtde_control.freedriveMode()
        raise NotImplementedError("freedrive function only written for ur5e")
        # TODO: if it's possible to manually push the base, great,
        # put that option here. if not, remove the above error throw,
        # document that there's no freedrive for the base here
        # and just put the arm to freedrive mode

    def unSetFreedrive(self):
        self._rtde_control.endFreedriveMode()
        raise NotImplementedError("freedrive function only written for ur5e")
        # TODO: if it's possible to manually push the base, great,
        # put that option here. if not, remove the above error throw,
        # document that there's no freedrive for the base here
        # and just put the arm to freedrive mode


class GazeboHeronRobotManager(AbstractHeronRobotManager, AbstractRealRobotManager):
    def __init__(self, args):
        super().__init__(args)
        if args.debug_prints:
            print("RealHeronRobotManager init")
        self._speed_slider = 1.0  # const
        self._v_cmd = np.zeros(self.model.nv)
        # raise NotImplementedError
        # TODO: instantiate topics for reading base position /ekf_something
        # TODO: instantiate topics for sending base velocity commands /cmd_vel

    def connectToGripper(self):
        pass

    def setInitialPose(self):
        # TODO: read position from localization topic,
        # put that into _q
        # HAS TO BE [x, y, cos(theta), sin(theta)] due to pinocchio's
        # representation of planar joint state
        self._q = pin.randomConfiguration(
                self.model, self.model.lowerPositionLimit, self.model.upperPositionLimit
            )
        self._q[0] = 0
        self._q[1] = 0
        self._q[2] = 1
        self._q[3] = 0

    def connectToRobot(self):
        pass

    def setSpeedSlider(self, value):
        pass

    # NOTE: handled by a topic callback
    def _updateQ(self):
        pass

    def _updateV(self):
        pass

    def _updateWrench(self):
        self._wrench_base = np.random.random(6)
        # NOTE: create a robot_math module, make this mapping a function called
        # mapse3ToDifferent frame or something like that
        mapping = np.zeros((6, 6))
        mapping[0:3, 0:3] = self._T_w_e.rotation
        mapping[3:6, 3:6] = self._T_w_e.rotation
        self._wrench = mapping.T @ self._wrench_base

    def zeroFtSensor(self):
        self._wrench_bias = np.zeros(6)

    def sendVelocityCommand(self, v):
        # speedj(qd, scalar_lead_axis_acc, hangup_time_on_command)
        assert type(v) == np.ndarray
        assert len(v) == self.model.nv
        # v_cmd_to_real = np.clip(v, -1 * self._max_v, self._max_v)
        # print(self._max_v)
        non_zero_mask = self._max_v != 0
        K = max(np.abs(v[non_zero_mask]) / np.abs(self._max_v[non_zero_mask]))
        v = v / max(1.0, K)
        v_cmd_to_real = np.clip(v, -1 * self._max_v, self._max_v)
        self._v_cmd = v_cmd_to_real
        # TODO: send the velocity command to the base by publishing to the
        # /vel_cmd topic
        # NOTE: look at sendVelocityCommand in RobotManager,
        # understand relationship between sendVelocityCommand and sendVelocityCommandToReal

        # NOTE: that sendVelocityCommand is overridden
        # in SingleArmWholeBodyInterface
        # (which AbstractHeronRobotManager inherits from) due
        # to different control modes resulting in different sizes
        # of qs and vs.
        # TODO: make sure that this function and they either work together,
        # or override again if they can't work toherther
    def sendVelocityCommandToReal(self, v):
        self.sendVelocityCommand(v)
        
    def stopRobot(self):
        self._v_cmd[:] = 0.0
        # raise NotImplementedError
        # TODO: we need to stop be the base as well.
        # option 1) send zero velocity commands.
        # but then make sure that it doesn't keep going forward
        # with a near-zero velocity (this happens on UR5e, that's why
        # the freedrive is started because it actually stops the arm)
        # option 2) programaticaly activate the emergency button
        # option 3) stop just the arm, do nothing for the base, and
        # clearly document that the robot has to be stopped by manually
        # pressing the emergency button

    def setFreedrive(self):
        pass
        # TODO: if it's possible to manually push the base, great,
        # put that option here. if not, remove the above error throw,
        # document that there's no freedrive for the base here
        # and just put the arm to freedrive mode

    def unSetFreedrive(self):
        pass

def heron_approximation() -> (
    tuple[pin.Model, pin.GeometryModel, pin.GeometryModel, pin.Data]
):
    # arm + gripper
    model_arm, collision_model_arm, visual_model_arm, _ = get_model(
        with_gripper_joints=True
    )

    # mobile base as planar joint (there's probably a better
    # option but whatever right now)
    model_mobile_base, geom_model_mobile_base = get_mobile_base_model(True)
    # frame-index should be 1
    model, visual_model = pin.appendModel(
        model_mobile_base,
        model_arm,
        geom_model_mobile_base,
        visual_model_arm,
        1,
        pin.SE3.Identity(),
    )
    model = pin.buildReducedModel(model, [8, 9], np.zeros(model.nq))
    data = model.createData()

    # fix gripper
    for geom in visual_model.geometryObjects:
        if "hand" in geom.name:
            s = geom.meshScale
            geom.meshcolor = np.array([1.0, 0.1, 0.1, 1.0])
            # this looks exactly correct lmao
            s *= 0.001
            geom.meshScale = s

    return model, visual_model.copy(), visual_model, data


# this gives me a flying joint for the camera,
# and a million joints for wheels -> it's unusable
# TODO: look what's done in pink, see if this can be usable
# after you've removed camera joint and similar.
# NOTE: NOT USED, BUT STUFF WILL NEED TO BE EXTRACTED FROM THIS EVENTUALLY
def get_heron_model_from_full_urdf() -> (
    tuple[pin.Model, pin.GeometryModel, pin.GeometryModel, pin.Data]
):

    # urdf_path_relative = files('smc.robot_descriptions.urdf').joinpath('ur5e_with_robotiq_hande_FIXED_PATHS.urdf')
    urdf_path_absolute = "/home/gospodar/home2/gospodar/lund/praxis/software/ros/ros-containers/home/model.urdf"
    # mesh_dir = files('smc')
    # mesh_dir_absolute = os.path.abspath(mesh_dir)
    mesh_dir_absolute = "/home/gospodar/lund/praxis/software/ros/ros-containers/home/heron_description/MIR_robot"

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

    data = pin.Data(model)

    return model, collision_model, visual_model, data

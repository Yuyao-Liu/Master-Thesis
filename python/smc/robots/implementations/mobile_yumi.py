from smc.robots.abstract_simulated_robotmanager import AbstractSimulatedRobotManager
from smc.robots.interfaces.whole_body_dual_arm_interface import (
    DualArmWholeBodyInterface,
)
from smc.robots.interfaces.mobile_base_interface import (
    get_mobile_base_model,
)
from smc.robots.implementations.yumi import get_yumi_model
from smc.robots.abstract_robotmanager import AbstractRobotManager

import pinocchio as pin
from argparse import Namespace
import numpy as np
from importlib.util import find_spec
if find_spec("rclpy"):
    from rclpy.time import Time
    from sensor_msgs.msg import JointState


class AbstractMobileYuMiRobotManager(DualArmWholeBodyInterface):

    def __init__(self, args):
        if args.debug_prints:
            print("MobileYuMiRobotManager init")
        self.args = args
        self._model, self._collision_model, self._visual_model, self._data = (
            get_mobile_yumi_model()
        )

        self._l_ee_frame_id = self._model.getFrameId("robl_tool0")
        self._r_ee_frame_id = self._model.getFrameId("robr_tool0")
        self._base_frame_id = self._model.getFrameId("mobile_base")
        # TODO: CHANGE THIS TO REAL VALUES
        self._MAX_ACCELERATION = 1.7  # const
        self._MAX_QD = 3.14  # const

        self._mode: DualArmWholeBodyInterface.control_mode = (
            DualArmWholeBodyInterface.control_mode.whole_body
        )
        self._comfy_configuration = np.array(
            [
                0.0,  # x
                0.0,  # y
                1.0,  # cos(theta)
                0.0,  # sin(theta)
                0.045,
                -0.155,
                -0.394,
                -0.617,
                -0.939,
                -0.343,
                -1.216,
                -0.374,
                -0.249,
                0.562,
                -0.520,
                0.934,
                -0.337,
                1.400,
            ]
        )
        # NOTE:  there's a shitton of stuff to re-write for this to work, and i'm not doing it now
        #        self._base_only_model, _ = get_mobile_base_model(underactuated=False)
        #        self._upper_body_model, _, _, _ = get_yumi_model()
        #        print(self._upper_body_model)

        super().__init__(args)

        # NOTE:  there's a shitton of stuff to re-write for this to work, and i'm not doing it now


#    @property
#    def model(self) -> pin.Model:
#        if self._mode == AbstractRobotManager.control_mode.whole_body:
#            return self._model
#        if self._mode == AbstractRobotManager.control_mode.base_only:
#            return self._base_only_model
#        if self._mode == AbstractRobotManager.control_mode.upper_body:
#            return self._upper_body_model
#        return self._model


class SimulatedMobileYuMiRobotManager(
    AbstractSimulatedRobotManager, AbstractMobileYuMiRobotManager
):
    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("SimulatedMobileYuMiRobotManager init")
        super().__init__(args)

    def setInitialPose(self):
        if self.args.start_from_current_pose:
            # TODO: add start from current pose for simulation, but actually read from the real robot
            # TODO: read position from localization topic,
            # put that into _q
            # HAS TO BE [x, y, cos(theta), sin(theta)] due to pinocchio's
            # representation of planar joint state
            raise NotImplementedError
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


class RealMobileYumiRobotManager(AbstractMobileYuMiRobotManager):
    def __init__(self, args):
        super().__init__(args)

    # TODO: here assert you need to have ros2 installed to run on real heron
    # and then set all this bullshit here instead of elsewhere
    def set_publisher_joints_cmd(self, publisher_joints_cmd):
        self.publisher_joints_cmd = publisher_joints_cmd
        print("set publisher_joints_cmd into RobotManager")

    def sendVelocityCommandToReal(self, v: np.ndarray):
        #        qd_base = qd[:3]
        #        qd_left = qd[3:10]
        #        qd_right = qd[10:]
        #        self.publisher_vel_base(qd_base)
        #        self.publisher_vel_left(qd_left)
        #        self.publisher_vel_right(qd_right)
        empty_msg = JointState()
        for i in range(29):
            empty_msg.velocity.append(0.0)
        msg = empty_msg
        msg.header.stamp = Time().to_msg()
        for i in range(3):
            msg.velocity[i] = v[i]
        for i in range(15, 29):
            msg.velocity[i] = v[i - 12]

        self.publisher_joints_cmd.publish(msg)

    # TODO: define set initial pose by reading it from the real robot (well, the appropriate ros2 topic in this case)
    def _updateQ(self):
        pass

    def _updateV(self):
        pass

    def stopRobot(self):
        self.sendVelocityCommand(np.zeros(self.model.nv))

    def setFreedrive(self):
        pass

    def unSetFreedrive(self):
        pass

    def connectToRobot(self):
        pass

    # TODO: create simulated gripper class to support the move, open, close methods - it's a mess now
    # in simulation we can just set the values directly
    def connectToGripper(self):
        pass


# TODO: define a separate mobile base for YuMi here
# necessary because the dimensions are not the same


def get_mobile_yumi_model() -> (
    tuple[pin.Model, pin.GeometryModel, pin.GeometryModel, pin.Data]
):

    model_arms, collision_model_arms, visual_model_arms, _ = get_yumi_model()
    model_mobile_base, geom_model_mobile_base = get_mobile_base_model(False)

    # frame-index should be 1
    model, visual_model = pin.appendModel(
        model_mobile_base,
        model_arms,
        geom_model_mobile_base,
        visual_model_arms,
        1,
        pin.SE3.Identity(),
    )
    data = model.createData()

    return model, visual_model.copy(), visual_model, data

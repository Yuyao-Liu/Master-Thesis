from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface
from smc.robots.abstract_simulated_robotmanager import AbstractSimulatedRobotManager

import pinocchio as pin
import hppfcl as fcl
import numpy as np
from argparse import Namespace


class AbstractMirRobotManager(MobileBaseInterface):
    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("AbstractMirRobotManager init")
        self._model, self._collision_model, self._visual_model, self._data = (
            mir_approximation()
        )
        self._base_frame_id = self.model.getFrameId("mobile_base")
        self._MAX_ACCELERATION = 1.7  # const
        self._MAX_QD = 3.14  # const
        super().__init__(args)


class SimulatedMirRobotManager(AbstractSimulatedRobotManager, AbstractMirRobotManager):
    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("SimulatedMirRobotManager init")
        super().__init__(args)

    def setInitialPose(self):
        self._q = np.zeros(4)
        self._q[0] = np.random.random()
        self._q[1] = np.random.random()
        theta = np.random.random() * 2 * np.pi - np.pi
        self._q[2] = np.cos(theta)
        self._q[3] = np.sin(theta)


class RealMirRobotManager(AbstractMirRobotManager):
    # TODO: implement
    def sendVelocityCommandToReal(self, v):
        """
        sendVelocityCommand
        -----
        different things need to be send depending on whether you're running a simulation,
        you're on a real robot, you're running some new simulator bla bla. this is handled
        here because this things depend on the arguments which are manager here (hence the
        class name RobotManager)
        """
        # y-direction is not possible on mir
        v[1] = 0
        cmd_msg = msg.Twist()
        cmd_msg.linear.x = v[0]
        cmd_msg.angular.z = v[2]
        # print("about to publish", cmd_msg)
        self.publisher_vel_base.publish(cmd_msg)
        # good to keep because updating is slow otherwise
        # it's not correct, but it's more correct than not updating
        # self.q = pin.integrate(self.model, self.q, qd * self.dt)
        raise NotImplementedError

    # TODO: implement
    def setInitialPose(self):
        raise NotImplementedError

    # TODO: implement
    def stopRobot(self):
        raise NotImplementedError

    # TODO: implement
    def setFreedrive(self):
        raise NotImplementedError

    # TODO: implement
    def unSetFreedrive(self):
        raise NotImplementedError


def mir_approximation() -> (
    tuple[pin.Model, pin.GeometryModel, pin.GeometryModel, pin.Data]
):
    # mobile base as planar joint (there's probably a better
    # option but whatever right now)
    model_mobile_base = pin.Model()
    model_mobile_base.name = "mobile_base"
    geom_model_mobile_base = pin.GeometryModel()
    joint_name = "mobile_base_planar_joint"
    parent_id = 0
    # TEST
    joint_placement = pin.SE3.Identity()
    MOBILE_BASE_JOINT_ID = model_mobile_base.addJoint(
        parent_id, pin.JointModelPlanar(), joint_placement.copy(), joint_name
    )
    # we should immediately set velocity limits.
    # there are no position limit by default and that is what we want.
    # TODO: put in heron's values
    # TODO: make these parameters the same as in mpc_params in the planner
    model_mobile_base.velocityLimit[0] = 2
    model_mobile_base.velocityLimit[1] = 0
    model_mobile_base.velocityLimit[2] = 2
    # TODO: i have literally no idea what reasonable numbers are here
    model_mobile_base.effortLimit[0] = 200
    model_mobile_base.effortLimit[1] = 0
    model_mobile_base.effortLimit[2] = 200
    # print("OBJECT_JOINT_ID",OBJECT_JOINT_ID)
    # body_inertia = pin.Inertia.FromBox(args.box_mass, box_dimensions[0],
    #        box_dimensions[1], box_dimensions[2])

    # pretty much random numbers
    # TODO: find heron (mir) numbers
    body_inertia = pin.Inertia.FromBox(30, 0.5, 0.3, 0.4)
    # maybe change placement to sth else depending on where its grasped
    model_mobile_base.appendBodyToJoint(
        MOBILE_BASE_JOINT_ID, body_inertia, pin.SE3.Identity()
    )
    box_shape = fcl.Box(0.5, 0.3, 0.4)
    body_placement = pin.SE3.Identity()
    geometry_mobile_base = pin.GeometryObject(
        "box_shape", MOBILE_BASE_JOINT_ID, box_shape, body_placement.copy()
    )

    geometry_mobile_base.meshColor = np.array([1.0, 0.1, 0.1, 0.3])
    geom_model_mobile_base.addGeometryObject(geometry_mobile_base)

    # have to add the frame manually
    # it's tool0 because that's used everywhere
    model_mobile_base.addFrame(
        pin.Frame(
            "mobile_base",
            MOBILE_BASE_JOINT_ID,
            0,
            joint_placement.copy(),
            pin.FrameType.JOINT,
        )
    )

    data = model_mobile_base.createData()

    return (
        model_mobile_base,
        geom_model_mobile_base.copy(),
        geom_model_mobile_base.copy(),
        data,
    )

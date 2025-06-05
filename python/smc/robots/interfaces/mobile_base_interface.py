from smc.robots.abstract_robotmanager import AbstractRobotManager

import numpy as np
import pinocchio as pin
from argparse import Namespace
import hppfcl as fcl


class MobileBaseInterface(AbstractRobotManager):
    """
    MobileBaseInterface
    -------------------
    This interface assumes that the first joint in the kinematic chain will be a planarJoint,
    modelling the mobile base of the robot.
    This does not exactly work for underactuated bases like differential drive,
    as some control inputs shouldn't even exist for this to be modelled properly.
    As it currently stands, the underactuation is enforced at the control level
    instead of the modelling level.
    One day this might be a "fully actuated base" class,
    implementing an abstract mobile base interface.
    """

    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("MobileBase init")
        self._base_frame_id: int
        self._T_w_b: pin.SE3
        if not hasattr(self, "_available_modes"):
            self._available_modes: list[AbstractRobotManager.control_mode] = [
                AbstractRobotManager.control_mode.whole_body,
                AbstractRobotManager.control_mode.base_only,  # in this case the same as wholebody
            ]
        super().__init__(args)

    @property
    def base_SE2_pose(self):
        # NOTE: idk if this is the best way to calculate theta
        # _q[:4] = [x, y, cos(theta), sin(theta)]
        theta = np.arccos(self._q[2])
        return np.array(list(self._q[:2]) + [theta])

    @property
    def base_frame_id(self):
        return self._base_frame_id

    @property
    def T_w_b(self):
        return self._T_w_b.copy()

    # NOTE: lil bit of evil to run some algorithms
    @property
    def T_w_e(self):
        return self._T_w_b.copy()

    def computeT_w_b(self, q: np.ndarray) -> pin.SE3:
        assert type(q) is np.ndarray
        pin.forwardKinematics(
            self.model,
            self.data,
            q,
        )
        # alternative if you want all frames
        # pin.updateFramePlacements(self.model, self.data)
        pin.updateFramePlacement(self.model, self.data, self._base_frame_id)
        return self.data.oMf[self._base_frame_id].copy()

    # TODO: make use of this for a standalone robot like the omnibot
    # just driving around with a full-body robot without using hands.
    # not prio right now and I don't want to deal with
    # how this interacts with other interfaces
    def forwardKinematics(self):
        pin.forwardKinematics(
            self.model,
            self.data,
            self._q,
        )
        pin.updateFramePlacement(self.model, self.data, self._base_frame_id)
        self._T_w_b = self.data.oMf[self._base_frame_id].copy()

    def _step(self):
        self._updateQ()
        self._updateV()
        self.forwardKinematics()

    def getJacobian(self) -> np.ndarray:
        # J = pin.computeFrameJacobian(
        #    self.model, self.data, self._q, self._base_frame_id
        # )
        # return J
        J_base = np.zeros((6, 3))
        J_base[:2, :2] = self.T_w_b.rotation[:2, :2]
        J_base[5, 2] = 1
        return J_base


def get_mobile_base_model(underactuated: bool) -> tuple[pin.Model, pin.GeometryModel]:

    # mobile base as planar joint (there's probably a better
    # option but whatever right now)
    model_mobile_base = pin.Model()
    model_mobile_base.name = "mobile_base"
    geom_model_mobile_base = pin.GeometryModel()
    joint_name = "mobile_base_planar_joint"
    parent_id = 0
    # TEST
    joint_placement = pin.SE3.Identity()
    # joint_placement.rotation = pin.rpy.rpyToMatrix(0, -np.pi/2, 0)
    # joint_placement.translation[2] = 0.2
    # TODO TODO TODO TODO TODO TODO TODO TODO
    # TODO: heron is actually a differential drive,
    # meaning that it is not a planar joint.
    # we could put in a prismatic + revolute joint
    # as the base (both joints being at same position),
    # and that should work for our purposes.
    # this makes sense for initial testing
    # because mobile yumi's base is a planar joint
    MOBILE_BASE_JOINT_ID = model_mobile_base.addJoint(
        parent_id, pin.JointModelPlanar(), joint_placement.copy(), joint_name
    )
    # we should immediately set velocity limits.
    # there are no position limit by default and that is what we want.
    # TODO: put in heron's values
    # TODO: make these parameters the same as in mpc_params in the planner
    if underactuated:
        model_mobile_base.velocityLimit[0] = 2
        # TODO: PUT THE CONSTRAINTS BACK!!!!!!!!!!!!!!!
        model_mobile_base.velocityLimit[1] = 0
        # model_mobile_base.velocityLimit[1] = 2
        model_mobile_base.velocityLimit[2] = 2
        # TODO: i have literally no idea what reasonable numbers are here
        model_mobile_base.effortLimit[0] = 200
        # TODO: PUT THE CONSTRAINTS BACK!!!!!!!!!!!!!!!
        model_mobile_base.effortLimit[1] = 0
        # model_mobile_base.effortLimit[1] = 2
        model_mobile_base.effortLimit[2] = 200
        # print("OBJECT_JOINT_ID",OBJECT_JOINT_ID)
        # body_inertia = pin.Inertia.FromBox(args.box_mass, box_dimensions[0],
        #        box_dimensions[1], box_dimensions[2])
    else:
        model_mobile_base.velocityLimit[0] = 2
        # TODO: PUT THE CONSTRAINTS BACK!!!!!!!!!!!!!!!
        model_mobile_base.velocityLimit[1] = 2
        # model_mobile_base.velocityLimit[1] = 2
        model_mobile_base.velocityLimit[2] = 2
        # TODO: i have literally no idea what reasonable numbers are here
        model_mobile_base.effortLimit[0] = 200
        # TODO: PUT THE CONSTRAINTS BACK!!!!!!!!!!!!!!!
        model_mobile_base.effortLimit[1] = 200
        # model_mobile_base.effortLimit[1] = 2
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
    box_shape = fcl.Box(0.8, 0.5, 0.872)
    body_placement = pin.SE3.Identity()
    body_placement.translation[2] += 0.436
    geometry_mobile_base = pin.GeometryObject(
        "box_shape", MOBILE_BASE_JOINT_ID, box_shape, body_placement.copy()
    )

    geometry_mobile_base.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model_mobile_base.addGeometryObject(geometry_mobile_base)
    arm2mir = pin.SE3.Identity()
    arm2mir.translation = np.array([-0.061854, -0.0045, 0.872])
    arm2mir.rotation = np.array([[0, -1, 0], [1, 0, 0], [0 ,0 ,1]])
    # have to add the frame manually
    model_mobile_base.addFrame(
        pin.Frame(
            "mobile_base",
            MOBILE_BASE_JOINT_ID,
            0,
            arm2mir.copy(),
            pin.FrameType.JOINT,
        )
    )

    return model_mobile_base, geom_model_mobile_base
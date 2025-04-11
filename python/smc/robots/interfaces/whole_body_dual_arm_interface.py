from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface
from smc.robots.interfaces.dual_arm_interface import DualArmInterface

import pinocchio as pin
from argparse import Namespace
import numpy as np

# from copy import deepcopy

# TODO: put back in when python3.12 will be safe to use
# from typing import override


class DualArmWholeBodyInterface(DualArmInterface, MobileBaseInterface):
    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("DualArmWholeBodyInterface init")
        self._available_modes: list[AbstractRobotManager.control_mode] = [
            AbstractRobotManager.control_mode.whole_body,
            AbstractRobotManager.control_mode.base_only,
            AbstractRobotManager.control_mode.upper_body,
            AbstractRobotManager.control_mode.left_arm_only,
            AbstractRobotManager.control_mode.right_arm_only,
        ]
        #        self._full_model = deepcopy(self._model)
        #        self._base_only_model = pin.buildReducedModel(self._model, [i for i in range(1, self._model.njoints + 1) if i > 1], np.zeros(self._model.nq))
        # NOTE: if you try to take out the mobile base joint, i.e. the first joint, i.e. a planarJoint, you get a segmentation fault :(
        # meaning this needs to be done on a case-by-case basis
        # also there's a shitton of stuff to re-write and i'm not doing it
        #        self._upper_body_model = pin.buildReducedModel(self._model, [i for i in range(1, self._model.njoints + 1) if i < 2], np.zeros(self._model.nq))
        super().__init__(args)

    #    @property
    #    def model(self) -> pin.Model:
    #        if self.control_mode == AbstractRobotManager.control_mode.whole_body:
    #            return self._full_model
    #        if self.control_mode == AbstractRobotManager.control_mode.base_only:
    #            return self._base_only_model
    #        if self.control_mode == AbstractRobotManager.control_mode.upper_body:
    #            return self._upper_body_model
    #        return self._full_model

    # TODO: override model property to produce the reduced version
    # depending on the control mode.
    # you might want to instantiate all of them in advance for easy switching later
    # NOTE: that this is currently only important for ocp construction,
    # even though it's obviously the correct move either way

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    @property
    def q(self) -> np.ndarray:
        if self._mode == self.control_mode.base_only:
            return self._q[:4]

        if self._mode == self.control_mode.upper_body:
            return self._q[4:]

        # NOTE: left should be on the left side of the joint values array
        if self._mode == self.control_mode.left_arm_only:
            return self._q[4 : (self.model.nq - 4) // 2]

        # NOTE: right should be on the right side of the joint values array
        if self._mode == self.control_mode.right_arm_only:
            return self._q[4 + (self.model.nq - 4) // 2 :]

        return self._q.copy()

    @property
    def nq(self):
        if self._mode == self.control_mode.base_only:
            return 4

        if self._mode == self.control_mode.upper_body:
            return self.model.nq - 4

        if self._mode == self.control_mode.left_arm_only:
            return (self.model.nq - 4) // 2

        if self._mode == self.control_mode.right_arm_only:
            return (self.model.nq - 4) // 2
        return self.model.nq

    @property
    def v(self) -> np.ndarray:
        if self._mode == self.control_mode.base_only:
            return self._v[:3]

        if self._mode == self.control_mode.upper_body:
            return self._v[3:]

        if self._mode == self.control_mode.left_arm_only:
            return self._v[3 : (self.model.nv - 3) // 2]

        if self._mode == self.control_mode.right_arm_only:
            return self._v[3 + (self.model.nv - 3) // 2 :]

        return self._v.copy()

    @property
    def nv(self) -> int:
        if self._mode == self.control_mode.base_only:
            return 3

        if self._mode == self.control_mode.upper_body:
            return self.model.nv - 3

        if self._mode == self.control_mode.left_arm_only:
            return (self.model.nv - 3) // 2

        if self._mode == self.control_mode.right_arm_only:
            return (self.model.nv - 3) // 2

        return self.model.nv

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    @property
    def max_v(self) -> np.ndarray:
        if self._mode == self.control_mode.base_only:
            return self._max_v[:3]
        if self._mode == self.control_mode.upper_body:
            return self._max_v[3:]
        if self._mode == self.control_mode.left_arm_only:
            return self._max_v[3 : 3 + (self.model.nv - 3) // 2]
        if self._mode == self.control_mode.right_arm_only:
            return self._max_v[3 + (self.model.nv - 3) // 2 :]
        return self._max_v.copy()

    # NOTE: lil' bit of evil to access cartesian controllers for single arm without changing the controller
    @property
    def T_w_e(self):
        if self.mode == self.control_mode.left_arm_only:
            return self._T_w_l.copy()
        if self.mode == self.control_mode.right_arm_only:
            return self._T_w_r.copy()
        if self.mode == self.control_mode.upper_body:
            return self._T_w_abs.copy()
        if self.mode == self.control_mode.base_only:
            return self._T_w_b.copy()
        return self._T_w_abs.copy()

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    def forwardKinematics(self) -> None:
        pin.forwardKinematics(
            self.model,
            self.data,
            self._q,
        )
        pin.updateFramePlacement(self.model, self.data, self._base_frame_id)
        self._T_w_b = self.data.oMf[self._base_frame_id].copy()
        pin.updateFramePlacement(self.model, self.data, self._l_ee_frame_id)
        pin.updateFramePlacement(self.model, self.data, self._r_ee_frame_id)
        self._T_w_l = self.data.oMf[self._l_ee_frame_id].copy()
        self._T_w_r = self.data.oMf[self._r_ee_frame_id].copy()
        self.T_w_abs  # will update _T_w_abs in the process

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    def getJacobian(self) -> np.ndarray:
        J_left_with_base = pin.computeFrameJacobian(
            self.model, self.data, self._q, self._l_ee_frame_id
        )[:, : 3 + (self.model.nv - 3) // 2]
        if self._mode == self.control_mode.left_arm_only:
            return J_left_with_base[:, 3:]

        # NOTE: the base jacobian can be extracted from either left or right frame -
        # since it's a body jacobian both have to be the same at the base.
        # for efficiency of course it would be best to construct it in place,
        # but who cares if it runs on time either way.
        if self._mode == self.control_mode.base_only:
            J_base = np.zeros((6, 3))
            J_base[:2, :2] = self.T_w_b.rotation[:2, :2]
            J_base[5, 2] = 1
            return J_base
            # return J_left_with_base[:, :3]

        J_right = pin.computeFrameJacobian(
            self.model, self.data, self._q, self._r_ee_frame_id
        )[:, 3 + (self.model.nv - 3) // 2 :]
        if self._mode == self.control_mode.right_arm_only:
            return J_right

        J_full = np.zeros((12, self.model.nv))
        J_full[:6, : 3 + (self.model.nv - 3) // 2] = J_left_with_base
        J_full[6:, 3 + (self.model.nv - 3) // 2 :] = J_right
        # NOTE: add base for right end-effector
        # look at note above returning base only jacobian
        J_full[6:, :3] = J_left_with_base[:, :3]

        if self._mode == self.control_mode.upper_body:
            return J_full[:, 3:]

        return J_full

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    def sendVelocityCommand(self, v_cmd) -> None:
        """
        sendVelocityCommand
        -------------------
        1) saturate the command to comply with hardware limits or smaller limits you've set
        2) send it via the particular robot's particular communication interface
        """
        assert type(v_cmd) == np.ndarray
        v_cmd_to_real = np.zeros(self.model.nv)
        if self._mode == self.control_mode.whole_body:
            v_cmd_to_real = v_cmd
        if self._mode == self.control_mode.base_only:
            v_cmd_to_real[:3] = v_cmd
        if self._mode == self.control_mode.upper_body:
            v_cmd_to_real[3:] = v_cmd
        if self._mode == self.control_mode.left_arm_only:
            v_cmd_to_real[3 : 3 + (self.model.nv - 3) // 2] = v_cmd
        if self._mode == self.control_mode.right_arm_only:
            v_cmd_to_real[3 + (self.model.nv - 3) // 2 :] = v_cmd

        v_cmd_to_real = np.clip(v_cmd_to_real, -1 * self._max_v, self._max_v)
        self.sendVelocityCommandToReal(v_cmd_to_real)

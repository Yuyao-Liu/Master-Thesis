from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.robots.abstract_robotmanager import AbstractRobotManager

import numpy as np
import pinocchio as pin
from argparse import Namespace


class DualArmInterface(SingleArmInterface):
    """
    DualArmInterface
    ----------------
    Provides the template for what the robot needs to have to use dual arm features.
    - l stands for left
    - r stands for right
    """

    # NOTE: you need to fill in the specific names from the urdf here for your robot
    # (better than putting in a magic number)
    # self._l_ee_frame_id = self.model.getFrameId("left_ee_name")
    # self._r_ee_frame_id = self.model.getFrameId("right_ee_name")

    def __init__(self, args: Namespace):
        self._T_w_abs: pin.SE3
        self._T_w_l: pin.SE3
        self._T_w_r: pin.SE3
        self._T_l_r: pin.SE3
        self._l_ee_frame_id: int
        self._r_ee_frame_id: int
        if args.debug_prints:
            print("DualArmInterface init")
        # this init might be called from wholebodyinterface in which case it has more modes,
        # and this would override those
        if not hasattr(self, "_available_modes"):
            self._available_modes: list[AbstractRobotManager.control_mode] = [
                AbstractRobotManager.control_mode.whole_body,
                AbstractRobotManager.control_mode.upper_body,  # in this case the same as wholebody
                AbstractRobotManager.control_mode.left_arm_only,
                AbstractRobotManager.control_mode.right_arm_only,
            ]
        super().__init__(args)

    @property
    def q(self) -> np.ndarray:
        # NOTE: left should be on the left side of the joint values array
        if self._mode == self.control_mode.left_arm_only:
            return self._q[: self.model.nq // 2]

        # NOTE: right should be on the right side of the joint values array
        if self._mode == self.control_mode.right_arm_only:
            return self._q[self.model.nq // 2 :]

        return self._q.copy()

    @property
    def nq(self):
        if self._mode == self.control_mode.left_arm_only:
            return self.model.nq // 2
        if self._mode == self.control_mode.right_arm_only:
            return self.model.nq // 2
        return self.model.nq

    @property
    def v(self) -> np.ndarray:
        if self._mode == self.control_mode.left_arm_only:
            return self._v[: self.model.nv // 2]
        if self._mode == self.control_mode.right_arm_only:
            return self._v[self.model.nv // 2 :]
        return self._v.copy()

    @property
    def nv(self):
        if self._mode == self.control_mode.left_arm_only:
            return self.model.nv // 2
        if self._mode == self.control_mode.right_arm_only:
            return self.model.nv // 2
        return self.model.nv

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    @property
    def max_v(self) -> np.ndarray:
        if self._mode == self.control_mode.left_arm_only:
            return self._max_v[: self.model.nv // 2]
        if self._mode == self.control_mode.right_arm_only:
            return self._max_v[self.model.nv // 2 :]
        return self._max_v.copy()

    @property
    def T_w_e(self):
        if self.mode == self.control_mode.left_arm_only:
            return self.T_w_l
        if self.mode == self.control_mode.right_arm_only:
            return self.T_w_r
        return self.T_w_abs

    @property
    def T_w_l(self):
        return self._T_w_l.copy()

    @property
    def l_ee_frame_id(self) -> int:
        return self._l_ee_frame_id

    @property
    def T_w_r(self):
        return self._T_w_r.copy()

    @property
    def r_ee_frame_id(self) -> int:
        return self._r_ee_frame_id

    # T_abs_l and T_abs_r are relative transformations between the absolute dual-arm frame,
    # and the left and right end-effector frames respectively
    @property
    def T_abs_r(self) -> pin.SE3:
        return self._T_w_abs.actInv(self._T_w_r)

    @property
    def T_abs_l(self) -> pin.SE3:
        return self._T_w_abs.actInv(self._T_w_l)

    @property
    def T_w_abs(self) -> pin.SE3:
        """
        getT_w_abs
        -----------
        get absolute frame, as seen in base frame, based on the current, or provident joint configuration
        """
        T_w_abs = pin.SE3.Interpolate(self._T_w_l, self._T_w_r, 0.5)
        self._T_w_abs = T_w_abs
        return self._T_w_abs.copy()

    def T_l_r(self) -> pin.SE3:
        T_l_r = self._T_w_l.actInv(self._T_w_r)
        self._T_l_r = T_l_r
        return self._T_l_r.copy()

    def getV_w_abs(self, V_w_l: pin.Motion, V_w_r: pin.Motion) -> pin.Motion:
        return 0.5 * (V_w_l + V_w_r)

    def getV_w_lr(self, V_w_l: pin.Motion, V_w_r: pin.Motion) -> pin.Motion:
        return V_w_r - V_w_l

    def getLeftRightT_w_e(self, q=None):
        """
        getLeftRightT_w_e
        -----------
        returns a tuple (T_w_l, T_w_r), i.e. left and right end-effector frames in the base frame,
        based on the current, or provident joint configuration
        """
        if q is None:
            return (self._T_w_l.copy(), self._T_w_r.copy())
        assert type(q) is np.ndarray
        # NOTE:
        # calling forward kinematics and updateFramePlacements here is ok
        # because we rely on robotmanager attributes instead of model.something in the rest of the code,
        # i.e. this won't update this class' atribute
        pin.forwardKinematics(
            self.model,
            self.data,
            q,
            #            np.zeros(self.model.nv),
            #            np.zeros(self.model.nv),
        )
        # NOTE: this also returns the frame, so less copying possible
        # pin.updateFramePlacements(self.model, self.data)
        pin.updateFramePlacement(self.model, self.data, self._l_ee_frame_id)
        pin.updateFramePlacement(self.model, self.data, self._r_ee_frame_id)
        return (
            self.data.oMf[self._l_ee_frame_id].copy(),
            self.data.oMf[self._r_ee_frame_id].copy(),
        )

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    def forwardKinematics(self) -> None:
        pin.forwardKinematics(
            self.model,
            self.data,
            self._q,
        )
        pin.updateFramePlacement(self.model, self.data, self._l_ee_frame_id)
        pin.updateFramePlacement(self.model, self.data, self._r_ee_frame_id)
        self._T_w_l = self.data.oMf[self._l_ee_frame_id].copy()
        self._T_w_r = self.data.oMf[self._r_ee_frame_id].copy()
        self.T_w_abs  # will update _T_w_abs in the process

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    # NOTE: this isn't useful unless it's in world frame
    def getJacobian(self) -> np.ndarray:
        # the other arm filled with zeros
        J_left = pin.computeFrameJacobian(
            self.model, self.data, self._q, self._l_ee_frame_id
        )[:, : self.model.nv // 2]
        if self._mode == self.control_mode.left_arm_only:
            return J_left

        J_right = pin.computeFrameJacobian(
            self.model, self.data, self._q, self._r_ee_frame_id
        )[:, self.model.nv // 2 :]
        if self._mode == self.control_mode.right_arm_only:
            return J_right

        J = np.zeros((12, self.nv))
        J[:6, : self.model.nv // 2] = J_left
        J[6:, self.model.nv // 2 :] = J_right
        return J

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

        if self._mode == self.control_mode.whole_body:
            v_cmd_to_real = v_cmd
        else:
            v_cmd_to_real = np.zeros(self.model.nv)

        if self._mode == self.control_mode.left_arm_only:
            v_cmd_to_real[: self.model.nv // 2] = v_cmd

        if self._mode == self.control_mode.right_arm_only:
            v_cmd_to_real[self.model.nv // 2 :] = v_cmd

        v_cmd_to_real = np.clip(v_cmd_to_real, -1 * self._max_v, self._max_v)
        self.sendVelocityCommandToReal(v_cmd_to_real)

    # NOTE: we almost certainly want to compute this w.r.t. T_w_abs
    # but i don't have time to write all the math out
    def computeManipulabilityIndexQDerivative(self) -> np.ndarray:

        def oneArm(joint_id: int) -> np.ndarray:
            J = pin.computeJointJacobian(self.model, self.data, self._q, joint_id)
            Jp = J.T @ np.linalg.inv(
                J @ J.T + np.eye(J.shape[0], J.shape[0]) * self.args.tikhonov_damp
            )
            # res = np.zeros(self.nv)
            # v0 = np.zeros(self.nv)
            res = np.zeros(self.model.nv)
            v0 = np.zeros(self.model.nv)
            for k in range(6):
                pin.computeForwardKinematicsDerivatives(
                    self.model,
                    self.data,
                    self._q,
                    Jp[:, k],
                    v0,
                    # self.model,
                    # self.data,
                    # self._q,
                    # v0,
                    # np.zeros(self.model.nv),
                )
                JqJpk = pin.getJointVelocityDerivatives(
                    self.model, self.data, joint_id, pin.LOCAL
                )[0]
                res += JqJpk[k, :]
            res *= self.computeManipulabilityIndex()
            return res

        l_joint_index = self.model.frames[self._l_ee_frame_id].parentJoint
        r_joint_index = self.model.frames[self._r_ee_frame_id].parentJoint

        # TODO: joint_ids obviously have to be defined per robot, this is a dirty hack
        # because i know i'm on yumi
        res_left = oneArm(l_joint_index)
        if self._mode == AbstractRobotManager.control_mode.left_arm_only:
            return res_left[:l_joint_index]
        res_right = oneArm(r_joint_index)
        if self._mode == AbstractRobotManager.control_mode.right_arm_only:
            return res_right[l_joint_index:]

        return res_left + res_right

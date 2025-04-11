from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface
from smc.robots.interfaces.single_arm_interface import SingleArmInterface

import pinocchio as pin
from argparse import Namespace
import numpy as np

# from copy import deepcopy

# TODO: put back in when python3.12 will be safe to use
# from typing import override


class SingleArmWholeBodyInterface(SingleArmInterface, MobileBaseInterface):
    """
    SingleArmWholeBodyInterface
    ---------------------------
    exists to provide either:
    1) whole body values
    2) base only value
    3) arm only value

    what you get depends on the mode you set - they're enumerate as above
    """

    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("SingleArmWholeBodyInterface init")
        self._mode: AbstractRobotManager.control_mode
        self._available_modes: list[AbstractRobotManager.control_mode] = [
            AbstractRobotManager.control_mode.whole_body,
            AbstractRobotManager.control_mode.base_only,
            AbstractRobotManager.control_mode.upper_body,
        ]
        super().__init__(args)

    # TODO: override model property to produce the reduced version
    # depending on the control mode.
    # you might want to instantiate all of them in advance for easy switching later
    # NOTE: that this is currently only important for ocp construction,
    # even though it's obviously the correct move either way

    #    @AbstractRobotManager.mode.setter
    #    def mode(self, mode: AbstractRobotManager.control_mode) -> None:
    #        assert type(mode) in self._available_modes
    #        self._mode = mode

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    @property
    def q(self) -> np.ndarray:
        if self._mode == self.control_mode.base_only:
            return self._q[:4]

        if self._mode == self.control_mode.upper_body:
            return self._q[4:]

        return self._q.copy()

    @property
    def nq(self):
        if self._mode == self.control_mode.base_only:
            return 4

        if self._mode == self.control_mode.upper_body:
            return self.model.nq - 4

        return self.model.nq

    @property
    def v(self) -> np.ndarray:
        if self._mode == self.control_mode.base_only:
            return self._v[:3]

        if self._mode == self.control_mode.upper_body:
            return self._v[3:]

        return self._v.copy()

    @property
    def nv(self):
        if self._mode == self.control_mode.base_only:
            return 3

        if self._mode == self.control_mode.upper_body:
            return self.model.nv - 3

        return self.model.nv

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    @property
    def max_v(self) -> np.ndarray:
        if self._mode == self.control_mode.base_only:
            return self._max_v[:3]
        if self._mode == self.control_mode.upper_body:
            return self._max_v[3:]
        return self._max_v.copy()

    # NOTE: lil' bit of evil to access cartesian controllers just for the base without changing the controller
    @property
    def T_w_e(self):
        if self.mode == self.control_mode.upper_body:
            return self._T_w_e.copy()
        if self.mode == self.control_mode.base_only:
            return self._T_w_b.copy()
        return self._T_w_e.copy()

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    def forwardKinematics(self) -> None:
        pin.forwardKinematics(
            self.model,
            self.data,
            self._q,
        )
        pin.updateFramePlacement(self.model, self.data, self._ee_frame_id)
        pin.updateFramePlacement(self.model, self.data, self._base_frame_id)
        self._T_w_e = self.data.oMf[self._ee_frame_id].copy()
        self._T_w_b = self.data.oMf[self._base_frame_id].copy()

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    def getJacobian(self) -> np.ndarray:
        J_full = pin.computeFrameJacobian(
            self.model, self.data, self._q, self._ee_frame_id
        )
        if self._mode == self.control_mode.base_only:
            return J_full[:, :3]

        if self._mode == self.control_mode.upper_body:
            return J_full[:, 3:]

        return J_full

    # TODO: put back in when python3.12 will be safe to use
    #    @override
    def sendVelocityCommand(self, v) -> None:
        """
        sendVelocityCommand
        -------------------
        1) saturate the command to comply with hardware limits or smaller limits you've set
        2) send it via the particular robot's particular communication interface
        """
        assert type(v) == np.ndarray

        if self._mode == self.control_mode.base_only:
            v = np.hstack((v, np.zeros(self.model.nv - 3)))

        if self._mode == self.control_mode.upper_body:
            v = np.hstack((np.zeros(3), v))

        assert len(v) == self.model.nv
        v = np.clip(v, -1 * self._max_v, self._max_v)
        self.sendVelocityCommandToReal(v)

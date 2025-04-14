import numpy as np
import pinocchio as pin
from smc.robots.abstract_robotmanager import AbstractRobotManager


class SingleArmInterface(AbstractRobotManager):
    def __init__(self, args):
        if args.debug_prints:
            print("SingleArmInterface init")
        self._ee_frame_id: int
        self._T_w_e: pin.SE3
        # this init might be called from wholebodyinterface in which case it has more modes,
        # and this would override those
        if not hasattr(self, "_available_modes"):
            self._available_modes: list[AbstractRobotManager.control_mode] = [
                AbstractRobotManager.control_mode.whole_body,
                AbstractRobotManager.control_mode.upper_body,  # in this case the same as wholebody
            ]
        super().__init__(args)

    @property
    def ee_frame_id(self):
        return self._ee_frame_id

    @property
    def T_w_e(self):
        return self._T_w_e.copy()

    def computeT_w_e(self, q) -> pin.SE3:
        assert type(q) is np.ndarray
        pin.forwardKinematics(
            self.model,
            self.data,
            q,
        )
        # alternative if you want all frames
        # pin.updateFramePlacements(self.model, self.data)
        pin.updateFramePlacement(self.model, self.data, self._ee_frame_id)
        return self.data.oMf[self._ee_frame_id].copy()

    def getJacobian(self) -> np.ndarray:
        return pin.computeFrameJacobian(
            self.model, self.data, self._q, self._ee_frame_id,
        )

    def forwardKinematics(self):
        pin.forwardKinematics(
            self.model,
            self.data,
            self._q,
        )
        pin.updateFramePlacement(self.model, self.data, self._ee_frame_id)
        self._T_w_e = self.data.oMf[self._ee_frame_id].copy()

    def _step(self):
        self._updateQ()
        self._updateV()
        self.forwardKinematics()

    # NOTE: manipulability calculations are here
    # only because i have no idea where to put them at the moment
    # TODO: put them in a better place

    def computeManipulabilityIndex(self) -> np.ndarray:
        J = self.getJacobian()
        return np.sqrt(np.linalg.det(J @ J.T))

    def computeManipulabilityIndexQDerivative(self) -> np.ndarray:
        joint_index = self.model.frames[self._ee_frame_id].parentJoint
        J = pin.computeJointJacobian(self.model, self.data, self._q, joint_index)
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
                self.model, self.data, joint_index, pin.LOCAL
            )[0]
            res += JqJpk[k, :]
        res *= self.computeManipulabilityIndex()
        return res

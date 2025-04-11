from smc.robots.abstract_robotmanager import AbstractRealRobotManager

import pinocchio as pin
import numpy as np


# TODO: rename this to pinocchio simulation somehow to account for the fact that this is pinocchio simulation.
# it could be in pybullet, mujoco or wherever else
class AbstractSimulatedRobotManager(AbstractRealRobotManager):
    def __init__(self, args):
        if args.debug_prints:
            print("AbstractSimulatedRobotManager init")
        super().__init__(args)

    # NOTE: can be overriden by any particular robot
    def setInitialPose(self):
        self._q = pin.randomConfiguration(
            self.model, self.model.lowerPositionLimit, self.model.upperPositionLimit
        )

    def sendVelocityCommandToReal(self, v):
        """
        sendVelocityCommand
        -----
        in simulation we just integrate the velocity for a dt and that's it
        """
        self._v = v
        # NOTE: we update joint angles here, and _updateQ does nothing (there is no communication)
        self._q = pin.integrate(self.model, self._q, v * self._dt)

    def _updateQ(self):
        pass

    def _updateV(self):
        pass

    # NOTE: simulation magic - it just stops immediatelly.
    # if you want a more realistic simulation, use an actual physics simulator
    def stopRobot(self):
        self._v = np.zeros(self.model.nv)

    # NOTE: since we're just integrating, nothign should happen here.
    # if this was in a physics simulator, you might want to run compliant control here instead
    def setFreedrive(self):
        pass

    # NOTE: since we're just integrating, nothign should happen here.
    # if this was in a physics simulator, you might want to return form compliant control
    # to whatever else instead
    def unSetFreedrive(self):
        pass

    # NOTE: this is pointless here, but in the case of a proper simulation you'd start the simulator,
    # or verify that it's running or something
    def connectToRobot(self):
        pass

    # TODO: create simulated gripper class to support the move, open, close methods - it's a mess now
    # in simulation we can just set the values directly
    def connectToGripper(self):
        pass

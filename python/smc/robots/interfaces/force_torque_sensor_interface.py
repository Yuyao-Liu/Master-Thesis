from smc.robots.interfaces.single_arm_interface import SingleArmInterface

import abc
import numpy as np
import time


class ForceTorqueSensorInterface(abc.ABC):
    def __init__(self):
        # in base frame
        # TODO: update so that __init__ accepts args
        #        if args.debug_prints:
        #            print("ForceTorqueSensorInterface init")
        self._wrench_base: np.ndarray = np.zeros(6)
        self._wrench: np.ndarray = np.zeros(6)
        # NOTE: wrench bias will be defined in the frame your sensor's gives readings
        self._wrench_bias: np.ndarray = np.zeros(6)

    # get this via model.getFrameId("frame_name")
    @property
    @abc.abstractmethod
    def ft_sensor_frame_id(self) -> int: ...

    # NOTE: ideally we can specify that this array is of length 6, but what can you do
    @property
    def wrench(self) -> np.ndarray:
        """
        getWrench
        ---------
        returns UNFILTERED! force-torque sensor reading, i.e. wrench,
        in the your prefered frame
        NOTE: it is given in the end-effector frame by default
        """
        return self._wrench.copy()

    # NOTE: ideally we can specify that this array is of length 6, but what can you do
    @property
    def wrench_base(self) -> np.ndarray:
        """
        getWrench
        ---------
        returns UNFILTERED! force-torque sensor reading, i.e. wrench,
        in the base frame of the robot
        """
        return self._wrench_base.copy()

    @abc.abstractmethod
    def _updateWrench(self) -> None:
        """
        get wrench reading from whatever interface you have.
        NOTE:
        1) it is YOUR job to know in which frame the reading is, and to map this
           BOTH to the base frame of the robot AND your prefered frame.
           this is because it is impossible to know what's the default on your sensor or where it is.
           this way you are forced to verify and know in which frame you get your readings from.
        2) do NOT do any filtering in this method - that happens elsewhere -
           here we just want the reading in correct frames
        3) you HAVE TO include this function in an overriden _step function
        4) if self.args.real == False, provide noise for better simulation testing
        """
        pass

    @abc.abstractmethod
    def zeroFtSensor(self) -> None:
        pass

    def calibrateFT(self, dt) -> None:
        """
        calibrateFT
        -----------
        TODO: make generic
        Read from the f/t sensor a bit, average the results
        and return the result.
        This can be used to offset the bias of the f/t sensor.
        NOTE: this is not an ideal solution.
        ALSO TODO: test whether the offset changes when
        the manipulator is in different poses.
        """
        ft_readings = []
        print("Will read from f/t sensors for a some number of seconds")
        print("and give you the average.")
        print("Use this as offset.")
        # NOTE: zeroFtSensor() needs to be called frequently because it drifts
        # by quite a bit in a matter of minutes.
        # if you are running something on the robot for a long period of time, you need
        # to reapply zeroFtSensor() to get reasonable results.
        # because the robot needs to stop for the zeroing to make sense,
        # this is the responsibility of the user!!!
        self.zeroFtSensor()
        for _ in range(2000):
            start = time.time()
            # ft = self._updateWrench()
            self._updateWrench()
            ft_readings.append(self._wrench_base.copy())
            end = time.time()
            diff = end - start
            if diff < dt:
                time.sleep(dt - diff)

        ft_readings = np.array(ft_readings)
        self._wrench_bias = np.average(ft_readings, axis=0)
        print("The wrench bias is:", self._wrench_bias)


class ForceTorqueOnSingleArmWrist(SingleArmInterface, ForceTorqueSensorInterface):
    def __init__(self, args):
        if args.debug_prints:
            print("ForceTorqueOnSingleArmWrist init")
        ForceTorqueSensorInterface.__init__(args)
        super().__init__(args)

    @property
    def ft_sensor_frame_id(self) -> int:
        return self._ee_frame_id

    def _step(self):
        super()._step()
        self._updateWrench()

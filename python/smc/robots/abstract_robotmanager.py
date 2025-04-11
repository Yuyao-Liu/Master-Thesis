from smc.visualization.visualizer import manipulatorVisualizer
from smc.logging.logger import Logger
from smc.multiprocessing.process_manager import ProcessManager
import abc
from argparse import Namespace
import pinocchio as pin
import numpy as np
from functools import partial
from enum import Enum


class AbstractRobotManager(abc.ABC):
    """
    RobotManager:
    ---------------
    - design goal: rely on pinocchio as much as possible while
                   concealing obvious bookkeeping
    - this class serves to:
        - store the non-interfacing information about the robot
          like maximum allowed velocity, pin.model etc
        - provide functions every robot has to have like getQ
    """

    # NOTE: these are all the possible modes for all possible robots.
    # all are declared here to enforce uniformity in child classes.
    # each robot has to declare which of these it supports.
    control_mode = Enum(
        "mode",
        [
            ("whole_body", 1),
            ("base_only", 2),
            ("upper_body", 3),
            ("left_arm_only", 4),
            ("right_arm_only", 5),
        ],
    )

    def __init__(self, args: Namespace):
        self._model: pin.Model
        self._data: pin.Data
        self._visual_model: pin.pinocchio_pywrap_default.GeometryModel
        self._collision_model: pin.pinocchio_pywrap_default.GeometryModel
        self._mode: AbstractRobotManager.control_mode = (
            AbstractRobotManager.control_mode.whole_body
        )
        self._available_modes: list[AbstractRobotManager.control_mode]
        self.args: Namespace = args
        if args.debug_prints:
            print("AbstractRobotManager init")
        ####################################################################
        #                    robot-related attributes                      #
        ####################################################################
        self.robot_name: str = args.robot  # const
        if self.args.ctrl_freq > 0:
            self._dt = 1 / self.args.ctrl_freq  # [s]
        else:
            # if we're going as fast as possible, we are in sim.
            # if we're in sim, we need to know for long to integrate,
            # and we set this to 1e-3 because there's absolutely no need to be more precise than that
            self._dt = 1e-3
        self._t: float = 0.0
        # NOTE: _MAX_ACCELERATION probably should be an array, but since UR robots only accept a float,
        #       we go for a float and pretend the every vector element is this 1 number
        # NOTE: this makes sense only for velocity-controlled robots.
        #       when torque control robot support will be introduce, we'll do something else here
        self._MAX_ACCELERATION: float | np.ndarray  # in joint space
        assert (
            self.args.acceleration <= self._MAX_ACCELERATION
            and self.args.acceleration > 0.0
        )
        self._acceleration = self.args.acceleration
        # _MAX_V is the robot's hardware joint velocity limit
        self._MAX_V: np.ndarray = self.model.velocityLimit
        # but you usually want to run at lower speeds for safety reasons
        assert self.args.max_v_percentage <= 1.0 and self.args.max_v_percentage > 0.0
        # so velocity commands are internally clipped to self._max_v
        self._max_v = np.clip(self._MAX_V, 0.0, args.max_v_percentage * self._MAX_V)
        # NOTE: make sure you've read pinocchio docs and understand why
        #       nq and nv are not necessarily the same number
        self._q = np.zeros(self.model.nq)
        self._v = np.zeros(self.model.nv)
        self._a = np.zeros(self.model.nv)

        self._comfy_configuration: np.ndarray

        # TODO: each robot should know which grippers are available
        # and set this there.
        self.gripper = None
        # initialize things that depend on q here
        # self._step()
        # check for where args.mode is legal is handled by argparse
        for mode in self.control_mode:
            if mode.name == args.mode:
                self.mode = mode

        ####################################################################
        #                    processes and utilities robotmanager owns     #
        ####################################################################
        # TODO: make a default
        self._log_manager: Logger
        if args.save_log:
            self._log_manager = Logger(args)

        # since there is only one robot and one visualizer, this is robotmanager's job
        # TODO: this should probably be transferred to a multiprocess manager,
        # or in whatever way be detangled from robots
        self.visualizer_manager: ProcessManager
        if self.args.visualizer:
            side_function = partial(
                manipulatorVisualizer,
                self.model,
                self.collision_model,
                self.visual_model,
            )
            # NOTE: it HAS TO be _q, we override q to support control of subsets of joints
            self.visualizer_manager = ProcessManager(
                args, side_function, {"q": self._q}, 0
            )
            # TODO: move this bs to visualizer, there are 0 reasons to keep it here
            if args.visualize_collision_approximation:
                # TODO: import the ellipses here, and write an update ellipse function
                # then also call that in controlloopmanager
                raise NotImplementedError(
                    "sorry, almost done - look at utils to get 90% of the solution!"
                )

    @property
    def mode(self) -> control_mode:
        return self._mode

    @mode.setter
    def mode(self, mode: control_mode) -> None:
        assert mode in self._available_modes
        self._mode = mode

    @property
    def model(self) -> pin.Model:
        return self._model

    @property
    def data(self) -> pin.Data:
        return self._data

    @property
    def visual_model(self) -> pin.pinocchio_pywrap_default.GeometryModel:
        return self._visual_model

    @property
    def collision_model(self) -> pin.pinocchio_pywrap_default.GeometryModel:
        return self._collision_model

    @property
    def max_v(self) -> np.ndarray:
        return self._max_v.copy()

    @abc.abstractmethod
    def setInitialPose(self) -> None:
        """
        setInitialPose
        --------------
        for example, if just integrating, do:
        self.q = pin.randomConfiguration(
            self.model, -1 * np.ones(self.model.nq), np.ones(self.model.nq)
        )
        NOTE: you probably want to specialize this for your robot
              to have it reasonable (no-self collisions, particular home etc)
        """
        pass

    @property
    def dt(self):
        return self._dt

    @property
    def q(self):
        return self._q.copy()

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def v(self):
        return self._v.copy()

    @property
    def nv(self) -> int:
        return self.model.nv

    # _updateQ and _updateV could be put into getters and setters of q and v respectively.
    # however, because i don't trust the interfaces of a random robot to play ball,
    # i want to keep this separated.
    # the point of q and v as properties is that you don't have to worry about using q or v when you like,
    # and this is not the case if a shitty interface creates lag upon every call.
    # if these functions are separate then i can guarantee that they are called once per control cycle in step
    # and i can know i used the shitty interface the least amount of times required - once per cycle.
    @abc.abstractmethod
    def _updateQ(self):
        """
        update internal _q joint angles variable with current angles obtained
        from your robot's communication interface
        """
        pass

    @abc.abstractmethod
    def _updateV(self):
        """
        update internal _v joint velocities variable with current velocities obtained
        from your robot's communication interface
        """
        pass

    @abc.abstractmethod
    def forwardKinematics(self):
        """
        forwardKinematics
        -----------------
        compute all the frames you care about based on current configuration
        and put them into class attributes.
        for a single manipulator with just the end-effector frame,
        this is a pointless function, but there is more to do for dual
        arm robots for example.

        This should be implemented by an interface.
        For example, for the SingleArmInterface you would do:

        def forwardKinematics(self):
            pin.forwardKinematics(self.model, self.data, self._q)
            pin.updateFramePlacement(self.model, self.data, self._ee_frame_id)
            self.T_w_e = self.data.oMf[self._ee_frame_id]
        """
        pass

    @abc.abstractmethod
    def _step(self):
        """
        _step
        ----
        Purpose:
        - update everything that should be updated on a step-by-step basis
        Reason for existance:
        - the problem this is solving is that you're not calling
          forwardKinematics, an expensive call, more than once per step.
        - you also don't need to remember to do call forwardKinematics
        Usage:
        - don't call this method yourself! this is CommandLoopManager's job
        - use getters to get copies of the variables you want

        Look at some interfaces to see particular implementations.
        """

    def sendVelocityCommand(self, v_cmd) -> None:
        """
        sendVelocityCommand
        -------------------
        1) saturate the command to comply with hardware limits or smaller limits you've set
        2) send it via the particular robot's particular communication interface
        """
        assert type(v_cmd) == np.ndarray
        assert len(v_cmd) == self.model.nv
        v_cmd_to_real = np.clip(v_cmd, -1 * self._max_v, self._max_v)
        self.sendVelocityCommandToReal(v_cmd_to_real)

    @abc.abstractmethod
    def sendVelocityCommandToReal(self, v_cmd_to_real): ...

    @abc.abstractmethod
    def stopRobot(self):
        """
        stopRobot
        ---------
        implement whatever stops the robot as cleanly as possible on your robot.
        does not need to be emergency option, could be sending zero velocity
        commands, could be switching to freedrive/leadthrough/whatever.

        depending on the communication interface on your robot,
        it could happen that it stays in "externally guided motion" mode,
        i.e. it keeps listening to new velocity commands, while executing the last velocity command.
        such a scenario is problematic because they last control command might not be
        all zeros, meaning the robot will keep slowly moving and hit something after 5min
        (obviously this exact thing happened to the author...)
        """
        pass

    @abc.abstractmethod
    def setFreedrive(self):
        """
        setFreedrive
        ------------
        set the robot into the "teaching mode", "freedrive mode", "leadthrough mode"
        or whatever the term is for when you can manually (with your hands) move the robot
        around
        """
        pass

    @abc.abstractmethod
    def unSetFreedrive(self):
        """
        setFreedrive
        ------------
        unset the robot into the "teaching mode", "freedrive mode", "leadthrough mode"
        or whatever the term is for when you can manually (with your hands) move the robot
        around, back to "wait" or "accept commands" or whatever mode the robot has to
        receive external commands.
        """
        pass

    # TODO: make faux gripper class to avoid this bullshit here
    def openGripper(self):
        if self.gripper is None:
            if self.args.debug_prints:
                print(
                    "you didn't select a gripper (no gripper is the default parameter) so no gripping for you"
                )
            return
        if (not self.args.simulation) and (not self.args.pinocchio_only):
            self.gripper.open()
        else:
            print("not implemented yet, so nothing is going to happen!")

    # TODO: make faux gripper class to avoid this bullshit here
    def closeGripper(self):
        if self.gripper is None:
            if self.args.debug_prints:
                print(
                    "you didn't select a gripper (no gripper is the default parameter) so no gripping for you"
                )
            return
        if (not self.args.simulation) and (not self.args.pinocchio_only):
            self.gripper.close()
        else:
            print("not implemented yet, so nothing is going to happen!")


    def getJacobian(self) -> np.ndarray:
        ...


    ########################################################################################
    # visualizer management. ideally transferred elsewhere
    ###################################################################################

    def killManipulatorVisualizer(self):
        """
        killManipulatorVisualizer
        ---------------------------
        if you're using the manipulator visualizer, you want to start it only once.
        because you start the meshcat server, initialize the manipulator and then
        do any subsequent changes with that server. there's no point in restarting.
        but this means you have to kill it manually, because the ControlLoopManager
        can't nor should know whether this is the last control loop you're running -
        RobotManager has to handle the meshcat server.
        and in this case the user needs to say when the tasks are done.
        """
        self.visualizer_manager.terminateProcess()

    def updateViz(self, viz_dict: dict):
        """
        updateViz
        ---------
        updates the viz and only the viz according to arguments
        NOTE: this function does not change internal variables!
        because it shouldn't - it should only update the visualizer
        """
        if self.args.visualizer:
            self.visualizer_manager.sendCommand(viz_dict)
        else:
            if self.args.debug_prints:
                print("you didn't select viz")

    def sendRectangular2DMapToVisualizer(self, map_as_list):
        for obstacle in map_as_list:
            length = obstacle[1][0] - obstacle[0][0]
            width = obstacle[3][1] - obstacle[0][1]
            height = 0.4  # doesn't matter because plan because planning is 2D
            pose = pin.SE3(
                np.eye(3),
                np.array(
                    [
                        obstacle[0][0] + (obstacle[1][0] - obstacle[0][0]) / 2,
                        obstacle[0][1] + (obstacle[3][1] - obstacle[0][1]) / 2,
                        0.0,
                    ]
                ),
            )
            dims = [length, width, height]
            command = {"obstacle_box": [pose, dims]}
            self.visualizer_manager.sendCommand(command)


class AbstractRealRobotManager(AbstractRobotManager, abc.ABC):
    """
    RobotManagerRealAbstract
    ------------------------
    enumerate the templates that the interface for your real robot
    has to have, connectToRobot
    """

    def __init__(self, args):
        super().__init__(args)
        if args.debug_prints:
            print("AbstractRealRobotManager init")
        self.connectToRobot()
        self.connectToGripper()
        self.setInitialPose()
        if self.args.visualizer:
            self.visualizer_manager.sendCommand(
                {
                    "q": self._q,
                }
            )

    @abc.abstractmethod
    def connectToRobot(self):
        """
        connectToRobot
        --------------
        create whatever is necessary to:
        1) send commands to the robot
        2) receive state information
        3) set options if applicable
        Setting up a gripper, force-torque sensor and similar is handled separately
        """
        pass

    @abc.abstractmethod
    def connectToGripper(self):
        """
        connectToGripper
        --------------
        create a gripper class based on selected argument.
        this needs to be made for each robot individually because even if the gripper
        is the same, the robot is likely to have its own way of interfacing it
        """
        pass

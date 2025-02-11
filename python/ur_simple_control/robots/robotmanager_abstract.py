import abc
import argparse
import pinocchio as pin
import numpy as np
from ur_simple_control.util.grippers.robotiq.robotiq_gripper import RobotiqGripper
from ur_simple_control.util.grippers.on_robot.twofg import TWOFG

from ur_simple_control.util.logging_utils import LogManager
from functools import partial
from ur_simple_control.visualize.visualize import manipulatorVisualizer

###################################
# TODO: make everything different on each robot an abstract method.
# TODO: try to retain non-robot defined functions like _step, getters, setters etc
# TODO: rename pinocchio_only to real (also toggle the boolean!)
############################################


class RobotManagerAbstract(abc.ABC):
    """
    RobotManager:
    ---------------
    - design goal: rely on pinocchio as much as possible while
                   concealing obvious bookkeeping
    - right now it is assumed you're running this on UR5e so some
      magic numbers are just put to it.
      this will be extended once there's a need for it.
    - it's just a boilerplate reduction class
    - if this was a real programming language, a lot of variables would really be private variables.
      as it currently stands, "private" functions have the '_' prefix
      while the public getters don't have a prefix.
    - TODO: write out default arguments needed here as well
    """

    # just pass all of the arguments here and store them as is
    # so as to minimize the amount of lines.
    # might be changed later if that seems more appropriate
    @abc.abstractmethod
    def __init__(self, args):
        self.args: argparse.Namespace  # const
        self.robot_name: str = args.robot  # const
        self.model: pin.Model  # const
        self.data: pin.Data
        self.visual_model: pin.pinocchio_pywrap_default.GeometryModel
        self.collision_model: pin.pinocchio_pywrap_default.GeometryModel

        if args.save_log:
            self._log_manager = LogManager(args)

        # TODO: add -1 option here meaning as fast as possible
        self._update_rate: int = self.args.ctrl_freq  # [Hz]
        self._dt = 1 / self._update_rate  # [s]
        self._t: float = 0.0
        # defined per robot - hardware limit
        self._MAX_ACCELERATION: float | np.ndarray  # in joint space
        assert (
            self.args.acceleration <= self._MAX_ACCELERATION
            and self.args.acceleration > 0.0
        )
        self._acceleration = self.args.acceleration
        # defined per robot - hardware limit
        self._MAX_QD: float | np.ndarray
        assert self.args.max_qd <= self._MAX_QD and self.args.max_qd > 0.0
        # internally clipped to this
        self._max_qd = args.max_qd

        # NOTE: make sure you've read pinocchio docs and understand why
        #       nq and nv are not necessarily the same number
        self.q = np.zeros(self.model.nq)
        self.v_q = np.zeros(self.model.nv)
        self.a_q = np.zeros(self.model.nv)

        self.gripper = None
        if (self.args.gripper != "none") and self.args.real:
            if self.args.gripper == "robotiq":
                self.gripper = RobotiqGripper()
                self.gripper.connect(args.robot_ip, 63352)
                self.gripper.activate()
            if self.args.gripper == "onrobot":
                self.gripper = TWOFG()

        # start visualize manipulator process if selected.
        # since there is only one robot and one visualizer, this is robotmanager's job
        self.visualizer_manager: None | ProcessManager
        if args.visualizer:
            side_function = partial(
                manipulatorVisualizer,
                self.model,
                self.collision_model,
                self.visual_model,
            )
            self.visualizer_manager = ProcessManager(
                args, side_function, {"q": self.q.copy()}, 0
            )
            if args.visualize_collision_approximation:
                # TODO: import the ellipses here, and write an update ellipse function
                # then also call that in controlloopmanager
                raise NotImplementedError(
                    "sorry, almost done - look at utils to get 90% of the solution!"
                )
        else:
            self.visualizer_manager = None

        self.connectToRobot()

        # wrench being the force-torque sensor reading, if any
        self.wrench_offset = np.zeros(6)

        self.setInitialPose()
        self._step()

    #######################################################################
    #               getters which assume you called step()                #
    #######################################################################

    # NOTE: just do nothing if you're only integrating with pinocchio,
    #       or start a physics simulator if you're using it
    @abc.abstractmethod
    def connectToRobot(self):
        pass

    @abc.abstractmethod
    def setInitialPose(self):
        """
        for example, if just integrating, do:
        self.q = pin.randomConfiguration(
            self.model, -1 * np.ones(self.model.nq), np.ones(self.model.nq)
        )
        NOTE: you probably want to specialize this for your robot
              to have it reasonable (no-self collisions, particular home etc)
        """
        pass

    def getQ(self):
        return self.q.copy()

    def getQd(self):
        return self.v_q.copy()

    def getT_w_e(self, q_given=None):
        if self.robot_name != "yumi":
            if q_given is None:
                return self.T_w_e.copy()
            else:
                assert type(q_given) is np.ndarray
                # calling these here is ok because we rely
                # on robotmanager attributes instead of model.something
                # (which is copying data, but fully separates state and computation,
                # which is important in situations like this)
                pin.forwardKinematics(
                    self.model,
                    self.data,
                    q_given,
                    np.zeros(self.model.nv),
                    np.zeros(self.model.nv),
                )
                # NOTE: this also returns the frame, so less copying possible
                # pin.updateFramePlacements(self.model, self.data)
                pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
                return self.data.oMf[self.ee_frame_id].copy()
        else:
            if q_given is None:
                return self.T_w_e_left.copy(), self.T_w_e_right.copy().copy()
            else:
                assert type(q_given) is np.ndarray
                # calling these here is ok because we rely
                # on robotmanager attributes instead of model.something
                # (which is copying data, but fully separates state and computation,
                # which is important in situations like this)
                pin.forwardKinematics(
                    self.model,
                    self.data,
                    q_given,
                    np.zeros(self.model.nv),
                    np.zeros(self.model.nv),
                )
                # NOTE: this also returns the frame, so less copying possible
                # pin.updateFramePlacements(self.model, self.data)
                pin.updateFramePlacement(self.model, self.data, self.r_ee_frame_id)
                pin.updateFramePlacement(self.model, self.data, self.l_ee_frame_id)
                return (
                    self.data.oMf[self.l_ee_frame_id].copy(),
                    self.data.oMf[self.r_ee_frame_id].copy(),
                )

    # this is in EE frame by default (handled in step which
    # is assumed to be called before this)
    def getWrench(self):
        return self.wrench.copy()

    def calibrateFT(self):
        """
        calibrateFT
        -----------
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
        self.rtde_control.zeroFtSensor()
        for i in range(2000):
            start = time.time()
            ft = self.rtde_receive.getActualTCPForce()
            ft_readings.append(ft)
            end = time.time()
            diff = end - start
            if diff < self.dt:
                time.sleep(self.dt - diff)

        ft_readings = np.array(ft_readings)
        self.wrench_offset = np.average(ft_readings, axis=0)
        print(self.wrench_offset)
        return self.wrench_offset.copy()

    def _step(self):
        """
        _step
        ----
        - the idea is to update everything that should be updated
          on a step-by-step basis
        - the actual problem this is solving is that you're not calling
          forwardKinematics, an expensive call, more than once per step.
        - within the TODO is to make all (necessary) variable private
          so that you can rest assured that everything is handled the way
          it's supposed to be handled. then have getters for these
          private variables which return deepcopies of whatever you need.
          that way the computations done in the control loop
          can't mess up other things. this is important if you want
          to switch between controllers during operation and have a completely
          painless transition between them.
          TODO: make the getQ, getQd and the rest here do the actual communication,
          and make these functions private.
          then have the deepcopy getters public.
          also TODO: make ifs for the simulation etc.
          this is less ifs overall right.
        """
        self._getQ()
        self._getQd()
        # self._getWrench()
        # computeAllTerms is certainly not necessary btw
        # but if it runs on time, does it matter? it makes everything available...
        # (includes forward kinematics, all jacobians, all dynamics terms, energies)
        # pin.computeAllTerms(self.model, self.data, self.q, self.v_q)
        pin.forwardKinematics(self.model, self.data, self.q, self.v_q)
        if self.robot_name != "yumi":
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            self.T_w_e = self.data.oMf[self.ee_frame_id].copy()
        else:
            pin.updateFramePlacement(self.model, self.data, self.l_ee_frame_id)
            pin.updateFramePlacement(self.model, self.data, self.r_ee_frame_id)
            self.T_w_e_left = self.data.oMf[self.l_ee_frame_id].copy()
            self.T_w_e_right = self.data.oMf[self.r_ee_frame_id].copy()
        # wrench in EE should obviously be the default
        self._getWrenchInEE(step_called=True)
        # this isn't real because we're on a velocity-controlled robot,
        # so this is actually None (no tau, no a_q, as expected)
        self.a_q = self.data.ddq
        # TODO NOTE: you'll want to do the additional math for
        # torque controlled robots here, but it's ok as is rn

    def setSpeedSlider(self, value):
        """
        setSpeedSlider
        ---------------
        update in all places
        """
        assert value <= 1.0 and value > 0.0
        if not self.args.pinocchio_only:
            self.rtde_io.setSpeedSlider(value)
        self.speed_slider = value

    def _getQ(self):
        """
        _getQ
        -----
        NOTE: private function for use in _step(), use the getter getQ()
        urdf treats gripper as two prismatic joints,
        but they do not affect the overall movement
        of the robot, so we add or remove 2 items to the joint list.
        also, the gripper is controlled separately so we'd need to do this somehow anyway
        NOTE: this gripper_past_pos thing is not working atm, but i'll keep it here as a TODO
        TODO: make work for new gripper
        """
        if not self.pinocchio_only:
            q = self.rtde_receive.getActualQ()
            if self.args.gripper == "robotiq":
                # TODO: make it work or remove it
                # self.gripper_past_pos = self.gripper_pos
                # this is pointless by itself
                self.gripper_pos = self.gripper.get_current_position()
                # the /255 is to get it dimensionless.
                # the gap is 5cm,
                # thus half the gap is 0.025m (and we only do si units here).
                q.append((self.gripper_pos / 255) * 0.025)
                q.append((self.gripper_pos / 255) * 0.025)
            else:
                # just fill it with zeros otherwise
                if self.robot_name == "ur5e":
                    q.append(0.0)
                    q.append(0.0)
            # let's just have both options for getting q, it's just a 8d float list
            # readability is a somewhat subjective quality after all
            q = np.array(q)
            self.q = q

    # TODO remove evil hack
    def _getT_w_e(self, q_given=None):
        """
        _getT_w_e
        -----
        NOTE: private function, use the getT_w_e() getter
        urdf treats gripper as two prismatic joints,
        but they do not affect the overall movement
        of the robot, so we add or remove 2 items to the joint list.
        also, the gripper is controlled separately so we'd need to do this somehow anyway
        NOTE: this gripper_past_pos thing is not working atm, but i'll keep it here as a TODO.
        NOTE: don't use this if use called _step() because it repeats forwardKinematics
        """
        test = True
        try:
            test = q_given.all() == None
            print(test)
            print(q_given)
        except AttributeError:
            test = True

        if test:
            if not self.pinocchio_only:
                q = self.rtde_receive.getActualQ()
                if self.args.gripper == "robotiq":
                    # TODO: make it work or remove it
                    # self.gripper_past_pos = self.gripper_pos
                    # this is pointless by itself
                    self.gripper_pos = self.gripper.get_current_position()
                    # the /255 is to get it dimensionless.
                    # the gap is 5cm,
                    # thus half the gap is 0.025m (and we only do si units here).
                    q.append((self.gripper_pos / 255) * 0.025)
                    q.append((self.gripper_pos / 255) * 0.025)
                else:
                    # just fill it with zeros otherwise
                    q.append(0.0)
                    q.append(0.0)
            else:
                q = self.q
        else:
            q = copy.deepcopy(q_given)
        q = np.array(q)
        self.q = q
        pin.forwardKinematics(self.model, self.data, q)
        if self.robot_name != "yumi":
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            self.T_w_e = self.data.oMf[self.ee_frame_id].copy()
        else:
            pin.updateFramePlacement(self.model, self.data, self.l_ee_frame_id)
            pin.updateFramePlacement(self.model, self.data, self.r_ee_frame_id)
            self.T_w_e_left = self.data.oMf[self.l_ee_frame_id].copy()
            self.T_w_e_right = self.data.oMf[self.r_ee_frame_id].copy()
            # NOTE: VERY EVIL, to bugfix things that depend on this like wrench (which we don't have on yumi)
            # self.T_w_e = self.data.oMf[self.l_ee_frame_id].copy()

    def _getQd(self):
        """
        _getQd
        -----
        NOTE: private function, use the _getQd() getter
        same note as _getQ.
        TODO NOTE: atm there's no way to get current gripper velocity.
        this means you'll probably want to read current positions and then finite-difference
        to get the velocity.
        as it stands right now, we'll just pass zeros in because I don't need this ATM
        """
        if not self.pinocchio_only:
            qd = self.rtde_receive.getActualQd()
            if self.args.gripper:
                # TODO: this doesn't work because we're not ensuring stuff is called
                # at every timestep
                # self.gripper_vel = (gripper.get_current_position() - self.gripper_pos) / self.dt
                # so it's just left unused for now - better give nothing than wrong info
                self.gripper_vel = 0.0
                # the /255 is to get it dimensionless
                # the gap is 5cm
                # thus half the gap is 0.025m and we only do si units here
                # no need to deepcopy because only literals are passed
                qd.append(self.gripper_vel)
                qd.append(self.gripper_vel)
            else:
                # just fill it with zeros otherwise
                qd.append(0.0)
                qd.append(0.0)
            # let's just have both options for getting q, it's just a 8d float list
            # readability is a somewhat subjective quality after all
            qd = np.array(qd)
            self.v_q = qd

    def _getWrenchRaw(self):
        """
        _getWrench
        -----
        different things need to be send depending on whether you're running a simulation,
        you're on a real robot, you're running some new simulator bla bla. this is handled
        here because this things depend on the arguments which are manager here (hence the
        class name RobotManager)
        """
        if not self.pinocchio_only:
            wrench = np.array(self.rtde_receive.getActualTCPForce())
        else:
            raise NotImplementedError("Don't have time to implement this right now.")

    def _getWrench(self):
        if not self.pinocchio_only:
            self.wrench = (
                np.array(self.rtde_receive.getActualTCPForce()) - self.wrench_offset
            )
        else:
            # TODO: do something better here (at least a better distribution)
            self.wrench = np.random.random(self.n_arm_joints)

    def _getWrenchInEE(self, step_called=False):
        if self.robot_name != "yumi":
            if not self.pinocchio_only:
                self.wrench = (
                    np.array(self.rtde_receive.getActualTCPForce()) - self.wrench_offset
                )
            else:
                # TODO: do something better here (at least a better distribution)
                self.wrench = np.random.random(self.n_arm_joints)
            if not step_called:
                self._getT_w_e()
            # NOTE: this mapping is equivalent to having a purely rotational action
            # this is more transparent tho
            mapping = np.zeros((6, 6))
            mapping[0:3, 0:3] = self.T_w_e.rotation
            mapping[3:6, 3:6] = self.T_w_e.rotation
            self.wrench = mapping.T @ self.wrench
        else:
            self.wrench = np.zeros(6)

    def sendQd(self, qd):
        """
        sendQd
        -----
        different things need to be send depending on whether you're running a simulation,
        you're on a real robot, you're running some new simulator bla bla. this is handled
        here because this things depend on the arguments which are manager here (hence the
        class name RobotManager)
        """
        # we're hiding the extra 2 prismatic joint shenanigans from the control writer
        # because there you shouldn't need to know this anyway
        if self.robot_name == "ur5e":
            qd_cmd = qd[:6]
            # np.clip is ok with bounds being scalar, it does what it should
            # (but you can also give it an array)
            qd_cmd = np.clip(qd_cmd, -1 * self.max_qd, self.max_qd)
            if not self.pinocchio_only:
                # speedj(qd, scalar_lead_axis_acc, hangup_time_on_command)
                self.rtde_control.speedJ(qd_cmd, self.acceleration, self.dt)
            else:
                # this one takes all 8 elements of qd since we're still in pinocchio
                # this is ugly, todo: fix
                qd = qd[:6]
                qd = qd_cmd.reshape((6,))
                qd = list(qd)
                qd.append(0.0)
                qd.append(0.0)
                qd = np.array(qd)
                self.v_q = qd
                self.q = pin.integrate(self.model, self.q, qd * self.dt)

        if self.robot_name == "heron":
            # y-direction is not possible on heron
            qd_cmd = np.clip(
                qd, -1 * self.model.velocityLimit, self.model.velocityLimit
            )
            # qd[1] = 0
            self.v_q = qd_cmd
            self.q = pin.integrate(self.model, self.q, qd_cmd * self.dt)

        if self.robot_name == "heronros":
            # y-direction is not possible on heron
            qd[1] = 0
            cmd_msg = msg.Twist()
            cmd_msg.linear.x = qd[0]
            cmd_msg.angular.z = qd[2]
            # print("about to publish", cmd_msg)
            self.publisher_vel_base.publish(cmd_msg)
            # good to keep because updating is slow otherwise
            # it's not correct, but it's more correct than not updating
            # self.q = pin.integrate(self.model, self.q, qd * self.dt)

        if self.robot_name == "mirros":
            # y-direction is not possible on heron
            qd[1] = 0
            cmd_msg = msg.Twist()
            cmd_msg.linear.x = qd[0]
            cmd_msg.angular.z = qd[2]
            # print("about to publish", cmd_msg)
            self.publisher_vel_base.publish(cmd_msg)
            # good to keep because updating is slow otherwise
            # it's not correct, but it's more correct than not updating
            # self.q = pin.integrate(self.model, self.q, qd * self.dt)

        if self.robot_name == "gripperlessur5e":
            qd_cmd = np.clip(qd, -1 * self.max_qd, self.max_qd)
            if not self.pinocchio_only:
                self.rtde_control.speedJ(qd_cmd, self.acceleration, self.dt)
            else:
                self.v_q = qd_cmd
                self.q = pin.integrate(self.model, self.q, qd_cmd * self.dt)

        if self.robot_name == "yumi":
            qd_cmd = np.clip(
                qd, -0.01 * self.model.velocityLimit, 0.01 * self.model.velocityLimit
            )
            self.v_q = qd_cmd
            #    if self.args.pinocchio_only:
            #        self.q = pin.integrate(self.model, self.q, qd_cmd * self.dt)
            #    else:
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
                msg.velocity[i] = qd_cmd[i]
            for i in range(15, 29):
                msg.velocity[i] = qd_cmd[i - 12]

            self.publisher_joints_cmd.publish(msg)

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

    #######################################################################
    #                          utility functions                          #
    #######################################################################

    def defineGoalPointCLI(self):
        """
        defineGoalPointCLI
        ------------------
        NOTE: this assume _step has not been called because it's run before the controlLoop
        --> best way to handle the goal is to tell the user where the gripper is
            in both UR tcp frame and with pinocchio and have them
            manually input it when running.
            this way you force the thinking before the moving,
            but you also get to view and analyze the information first
        TODO get the visual thing you did in ivc project with sliders also.
        it's just text input for now because it's totally usable, just not superb.
        but also you do want to have both options. obviously you go for the sliders
        in the case you're visualizing, makes no sense otherwise.
        """
        self._getQ()
        q = self.getQ()
        # define goal
        pin.forwardKinematics(self.model, self.data, np.array(q))
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        T_w_e = self.data.oMf[self.ee_frame_id]
        print("You can only specify the translation right now.")
        if not self.pinocchio_only:
            print(
                "In the following, first 3 numbers are x,y,z position, and second 3 are r,p,y angles"
            )
            print(
                "Here's where the robot is currently. Ensure you know what the base frame is first."
            )
            print(
                "base frame end-effector pose from pinocchio:\n",
                *self.data.oMi[6].translation.round(4),
                *pin.rpy.matrixToRpy(self.data.oMi[6].rotation).round(4)
            )
            print("UR5e TCP:", *np.array(self.rtde_receive.getActualTCPPose()).round(4))
        # remain with the current orientation
        # TODO: add something, probably rpy for orientation because it's the least number
        # of numbers you need to type in
        Mgoal = T_w_e.copy()
        # this is a reasonable way to do it too, maybe implement it later
        # Mgoal.translation = Mgoal.translation + np.array([0.0, 0.0, -0.1])
        # do a while loop until this is parsed correctly
        while True:
            goal = input(
                "Please enter the target end-effector position in the x.x,y.y,z.z format: "
            )
            try:
                e = "ok"
                goal_list = goal.split(",")
                for i in range(len(goal_list)):
                    goal_list[i] = float(goal_list[i])
            except:
                e = exc_info()
                print("The input is not in the expected format. Try again.")
                print(e)
            if e == "ok":
                Mgoal.translation = np.array(goal_list)
                break
        print("this is goal pose you defined:\n", Mgoal)

        # NOTE i'm not deepcopying this on purpose
        # but that might be the preferred thing, we'll see
        self.Mgoal = Mgoal
        if self.args.visualize_manipulator:
            # TODO document this somewhere
            self.visualizer_manager.sendCommand({"Mgoal": Mgoal})
        return Mgoal

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

    def stopRobot(self):
        if not self.args.pinocchio_only:
            print("stopping via freedrive lel")
            self.rtde_control.freedriveMode()
            time.sleep(0.5)
            self.rtde_control.endFreedriveMode()

    def setFreedrive(self):
        if self.robot_name in ["ur5e", "gripperlessur5e"]:
            self.rtde_control.freedriveMode()
        else:
            raise NotImplementedError("freedrive function only written for ur5e")

    def unSetFreedrive(self):
        if self.robot_name in ["ur5e", "gripperlessur5e"]:
            self.rtde_control.endFreedriveMode()
        else:
            raise NotImplementedError("freedrive function only written for ur5e")

    def updateViz(self, viz_dict: dict):
        """
        updateViz
        ---------
        updates the viz and only the viz according to arguments
        NOTE: this function does not change internal variables!
        because it shouldn't - it should only update the visualizer
        """
        if self.args.visualize_manipulator:
            self.visualizer_manager.sendCommand(viz_dict)
        else:
            if self.args.debug_prints:
                print("you didn't select viz")

    def set_publisher_vel_base(self, publisher_vel_base):
        self.publisher_vel_base = publisher_vel_base
        print("set vel_base_publisher into robotmanager")

    def set_publisher_vel_left(self, publisher_vel_left):
        self.publisher_vel_left = publisher_vel_left
        print("set vel_left_publisher into robotmanager")

    def set_publisher_vel_right(self, publisher_vel_right):
        self.publisher_vel_right = publisher_vel_right
        print("set vel_right_publisher into robotmanager")

    def set_publisher_joints_cmd(self, publisher_joints_cmd):
        self.publisher_joints_cmd = publisher_joints_cmd
        print("set publisher_joints_cmd into RobotManager")

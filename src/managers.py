# PYTHON_ARGCOMPLETE_OK
# TODO rename all private variables to start with '_'
# TODO: make importing nicer with __init__.py files
# put all managers into a managers folder,
# and have each manager in a separate file, this is getting ridiculous
import pinocchio as pin
import numpy as np
import time
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
from ur_simple_control.util.grippers.robotiq.robotiq_gripper import RobotiqGripper
from ur_simple_control.util.grippers.on_robot.twofg import TWOFG
import copy
import signal
from ur_simple_control.util.get_model import get_model, heron_approximation, heron_approximationDD, getGripperlessUR5e, mir_approximation, get_yumi_model
from collections import deque
from ur_simple_control.visualize.visualize import plotFromDict, realTimePlotter, manipulatorVisualizer
from ur_simple_control.util.logging_utils import LogManager
from multiprocessing import Process, Queue, Lock, shared_memory
# argcomplete is an external package which creates tab completion in shell
# argparse is argument parsing from the standard library
import argcomplete, argparse
from sys import exc_info
from types import NoneType
from os import getpid
from functools import partial
import pickle
import typing

# import ros stuff if you're rosing
# TODO: add more as needed
import importlib.util
if importlib.util.find_spec('rclpy'):
    from geometry_msgs import msg 
    from sensor_msgs.msg import JointState
    from rclpy.time import Time

"""
general notes
---------------
The first design principle of this library is to minimize the time needed
to go from a control algorithm on paper to the same control algorithm 
running on the real robot. The second design principle is to have
the code as simple as possible. In particular, this pertains to avoiding
overly complex abstractions and having the code as concise as possible.
The user is expected to read and understand the entire codebase because
changes will have to accomodate it to their specific project.
Target users are control engineers.
The final design choices are made to accommodate these sometimes opposing goals
to the best of the author's ability.

This file contains a robot manager and a control loop manager.
The point of these managers is to handle:
    - boiler plate code around the control loop which is always the same
    - have all the various parameters neatly organized and in one central location
    - hide the annoying if-elses of different APIs required 
      for the real robot and various simulations with single commands
      that just do exactly what you want them to do


current state
-------------
Everything is UR specific or supports pinocchio only simulation,
and only velocity-controlled robot functions exist.

long term vision
-------------------
Cut out the robot-specific parts out of the manager classes,
and create child classes for particular robots.
There is an interface to a physics simulator.
Functions for torque controlled robots exist.
"""

def getMinimalArgParser():
    """
    getDefaultEssentialArgs
    ------------------------
    returns a parser containing:
        - essential arguments (run in real or in sim)
        - parameters for (compliant)moveJ
        - parameters for (compliant)moveL
    """
    parser = argparse.ArgumentParser(description="Run something with \
            Simple Manipulator Control", \
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #################################################
    #  general arguments: connection, plotting etc  #
    #################################################
    parser.add_argument('--robot', type=str, \
            help="which robot you're running or simulating", default="ur5e", \
            choices=['ur5e', 'heron', 'heronros', 'gripperlessur5e', 'mirros', 'yumi'])
    parser.add_argument('--simulation', action=argparse.BooleanOptionalAction, \
            help="whether you are running the UR simulator", default=False)
    parser.add_argument('--robot-ip', type=str, 
            help="robot's ip address (only needed if running on the real robot)", \
                    default="192.168.1.102")
    parser.add_argument('--pinocchio-only', action=argparse.BooleanOptionalAction, \
            help="whether you want to just integrate with pinocchio", default=True)
    parser.add_argument('--ctrl-freq', type=int, \
            help="frequency of the control loop", default=500)
    parser.add_argument('--fast-simulation', action=argparse.BooleanOptionalAction, \
            help="do you want simulation to as fast as possible? (real-time viz relies on 500Hz)", default=False)
    parser.add_argument('--visualizer', action=argparse.BooleanOptionalAction, \
            help="whether you want to visualize the manipulator and workspace with meshcat", default=True)
    parser.add_argument('--plotter', action=argparse.BooleanOptionalAction, \
            help="whether you want to have some real-time matplotlib graphs (parts of log_dict you select)", default=True)
    parser.add_argument('--gripper', type=str, \
            help="gripper you're using (no gripper is the default)", \
                        default="none", choices=['none', 'robotiq', 'onrobot'])
    # TODO: make controlloop manager run in a while True loop and remove this
    # ==> max-iterations actually needs to be an option. sometimes you want to simulate and exit
    #     if the convergence does not happen in a reasonable amount of time,
    #     ex. goal outside of workspace has been passed or something
    # =======> if it's set to 0 then the loops run infinitely long
    parser.add_argument('--max-iterations', type=int, \
            help="maximum allowable iteration number (it runs at 500Hz)", default=100000)
    parser.add_argument('--speed-slider', type=float,\
            help="cap robot's speed with the speed slider \
                    to something between 0 and 1, 0.5 by default \
                    BE CAREFUL WITH THIS.", default=1.0)
    parser.add_argument("--start-from-current-pose", action=argparse.BooleanOptionalAction, \
            help="if connected to the robot, read the current pose and set it as the initial pose for the robot. \
                 very useful and convenient when running simulation before running on real", \
                         default=False)
    parser.add_argument('--acceleration', type=float, \
            help="robot's joints acceleration. scalar positive constant, max 1.7, and default 0.3. \
                   BE CAREFUL WITH THIS. the urscript doc says this is 'lead axis acceleration'.\
                   TODO: check what this means", default=0.3)
    parser.add_argument('--max-qd', type=float, \
            help="robot's joint velocities [rad/s]. scalar positive constant, max 3.14, and default 0.5. \
                   BE CAREFUL WITH THIS. also note that wrist joints can go to 6.28 rad, but here \
                        everything is clipped to this one number.", default=0.5)
    parser.add_argument('--debug-prints', action=argparse.BooleanOptionalAction, \
            help="print some debug info", default=False)
    parser.add_argument('--save-log', action=argparse.BooleanOptionalAction, \
            help="whether you want to save the log of the run. it saves \
                        what you pass to ControlLoopManager. check other parameters for saving directory and log name.", default=False)
    parser.add_argument('--save-dir', type=str, \
            help="path to where you store your logs. default is ./data, but if that directory doesn't exist, then /tmp/data is created and used.", \
            default='./data')
    parser.add_argument('--run-name', type=str, \
            help="name the whole run/experiment (name of log file). note that indexing of runs is automatic and under a different argument.", \
            default='latest_run')
    parser.add_argument('--index-runs', action=argparse.BooleanOptionalAction, \
            help="if you want more runs of the same name, this option will automatically assign an index to a new run (useful for data collection).", default=False)
    parser.add_argument('--past-window-size', type=int, \
            help="how many timesteps of past data you want to save", default=5)
    # maybe you want to scale the control signal (TODO: NOT HERE)
    parser.add_argument('--controller-speed-scaling', type=float, \
            default='1.0', help='not actually_used atm')
    ########################################
    #  environment interaction parameters  #
    ########################################
    parser.add_argument('--contact-detecting-force', type=float, \
            #default=1.3, help='the force used to detect contact (collision) in the moveUntilContact function')
            default=2.8, help='the force used to detect contact (collision) in the moveUntilContact function')
    parser.add_argument('--minimum-detectable-force-norm', type=float, \
            help="we need to disregard noise to converge despite filtering. \
                  a quick fix is to zero all forces of norm below this argument threshold.",
                 default=3.0)
    # TODO make this work without parsing (or make it possible to parse two times)
    #if (args.gripper != "none") and args.simulation:
    #    raise NotImplementedError('Did not figure out how to put the gripper in \
    #            the simulation yet, sorry :/ . You can have only 1 these flags right now')
    parser.add_argument('--visualize-collision-approximation', action=argparse.BooleanOptionalAction, \
            help="whether you want to visualize the collision approximation used in controllers with obstacle avoidance", default=True)
    return parser


class ControlLoopManager:
    """
    ControlLoopManager
    -------------------
    Slightly fancier programming (passing a function as argument and using functools.partial)
    to get a wrapper around the control loop.
    In other words, it's the book-keeping around the actual control loop.
    It's a class because it keeps non-directly-control-loop-related parameters
    like max_iterations, what data to save etc.
    NOTE: you give this the ready-made control loop.
    if it has arguments, bake them in with functools.partial.
    Handles short-term data saving and logging.
    Details on this are given below.

    Short term data saving:
            - it's a dictionaries of deques (initialized here), because deque is the most convenient class 
              for removing the first element and appending a last (it is just a linked list under the hood of course).
            - it's a dictionary for modularity's sake, because this way you can save whatever you want
            - and it will just work based on dictionary keys.
            - it is the user's resposibility to make sure they're providing correct data.
            - --> TODO but make an assert for the keys at least
            - in the c++ imlementation, make the user write their own struct or something.
            - since this is python, you need to give me initial values to infer types.
            - you need to provide initial values to populate the deque to start anyway.

    Logging data (for analysis and plotting):
            - it can only be handled here because the control loop itself only cares about the present/
              a small time-window around it.
            - saves it all in a dictionary of ndarrays (initialized here), returns that after a run
              TODO: it's provided by the user now, make it actually initialize here!!!
            - you need to specify which keys you're using to do the initialization 
            - later, the controlLoop needs to return what's to be save in a small temporary dict.
            - NOTE: this is of course a somewhat redundant solution, but it's the simplest
              and most flexible way of doing it. 
              it probably will be done some other way down the line, but it works and is not
              a priority right now

    Other info:
    - relies on RobotManager to handle all the magic numbers 
      that are not truly only control loop related

    """

    def __init__(self, robot_manager, controlLoop, args, save_past_item, log_item):
        signal.signal(signal.SIGINT, self.stopHandler)
        self.pid = getpid()
        self.max_iterations = args.max_iterations
        self.robot_manager = robot_manager
        self.controlLoop = controlLoop
        self.final_iteration = -1 # because we didn't even start yet
        self.args = args
        self.iter_n = 0
        self.past_data = {}
        # save_past_dict has to have the key and 1 example of what you're saving
        # so that it's type can be inferred (but we're in python so types don't really work).
        # the good thing is this way you also immediatelly put in the initial values
        for key in save_past_item:
            self.past_data[key] = deque()
            # immediatelly populate every deque with initial values
            for i in range(self.args.past_window_size):
                # deepcopy just in case, better safe than sorry plus it's during initialization,
                # not real time
                self.past_data[key].append(copy.deepcopy(save_past_item[key]))

        # similar story for log_dict as for past_data,
        # except this is not used in the control loop,
        # we don't predeclare sizes, but instead
        # just shove items into linked lists (python lists) in dictionaries (hash-maps)
        self.log_dict = {}
        for key in log_item:
            self.log_dict[key] = []

        if self.args.plotter:
            self.plotter_manager = ProcessManager(args, realTimePlotter, log_item, 0)


    def run_one_iter(self, i):
        """
        run
        ---
        do timing to run at 500Hz.
        also handle the number of iterations.
        it's the controlLoop's responsibility to break if it achieved it's goals.
        this is done via the breakFlag
        """
        # NOTE: all required pre-computations are handled here
        self.robot_manager._step()
        # TODO make the arguments to controlLoop kwargs or whatever
        # so that you don't have to declare them on client side if you're not using them
        breakFlag, latest_to_save_dict, log_item = self.controlLoop(i, self.past_data)
        self.final_iteration = i

        # update past rolling window
        # TODO: write an assert assuring the keys are what's been promised
        # (ideally this is done only once, not every time, so think whether/how that can be avoided)
        for key in latest_to_save_dict:
            # remove oldest entry
            self.past_data[key].popleft()
            # add new entry
            self.past_data[key].append(latest_to_save_dict[key])
        
        # log the data
        # check that you can
        # TODO only need to check this once, pls enforce better
        #if len(self.log_dict) > 0:
        for key in log_item:
                #if key not in self.log_dict.keys():
                #    raise KeyError("you need to provide log items you promised!")
                #    break
            self.log_dict[key].append(log_item[key])
        
        # TODO: do it this way if running on the real robot.
        # but if not, we want to control the speed of the simulation,
        # and how much we're plotting.
        # so this should be an argument that is use ONLY if we're in simulation
        if i % 20 == 0:
            # don't send what wasn't ready
            if self.args.visualizer:
                if self.robot_manager.robot_name != "yumi":
                    self.robot_manager.visualizer_manager.sendCommand({"q" : self.robot_manager.q,
                                                          "T_w_e" : self.robot_manager.getT_w_e()})
                else:
                    T_w_e_left, T_w_e_right = self.robot_manager.getT_w_e()
                    self.robot_manager.visualizer_manager.sendCommand({"q" : self.robot_manager.q,
                                                          "T_w_e" : T_w_e_left})
                if self.robot_manager.robot_name == "heron":
                    T_base = self.robot_manager.data.oMi[1]
                    self.robot_manager.visualizer_manager.sendCommand({"T_base" : T_base})
#                if self.robot_manager.robot_name == "yumi":
#                    T_base = self.robot_manager.data.oMi[1]
#                    self.robot_manager.visualizer_manager.sendCommand({"T_base" : T_base})

                if self.args.visualize_collision_approximation:
                    pass
                # TODO: here call robot manager's update ellipses function

            if self.args.plotter:
                # don't put new stuff in if it didn't handle the previous stuff.
                # it's a plotter, who cares if it's late. 
                # the number 5 is arbitrary
                self.plotter_manager.sendCommand(log_item)
        return breakFlag

    def run(self):
        self.final_iteration = 0
        for i in range(self.max_iterations):
            start = time.time()
            breakFlag = self.run_one_iter(i)

            # break if done
            if breakFlag:
                break

            # sleep for the rest of the frequency cycle
            end = time.time()
            diff = end - start
            if self.robot_manager.dt < diff:
                if self.args.debug_prints:
                    print("missed deadline by", diff - self.robot_manager.dt)
                continue
            else:
                # there's no point in sleeping if we're in simulation
                # NOTE: it literally took me a year to put this if here
                # and i have no idea why i didn't think of it before lmao 
                # (because i did know about it, just didn't even think of changing it)
                if not (self.args.pinocchio_only and self.args.fast_simulation):
                    time.sleep(self.robot_manager.dt - diff)

        ######################################################################
        # for over
        ######################################################################
        if self.args.plotter:
            self.plotter_manager.terminateProcess()
        if self.args.save_log:
            self.robot_manager.log_manager.storeControlLoopRun(self.log_dict, self.controlLoop.func.__name__, self.final_iteration)
        if i < self.max_iterations -1:
            if self.args.debug_prints:
                print("success in", i, "iterations!")
        else:
            print("FAIL: did not succed in", self.max_iterations, "iterations")
            #self.stopHandler(None, None)

    def stopHandler(self, signum, frame):
        """
        stopHandler
        -----------
        upon receiving SIGINT it sends zeros for speed commands to
        stop the robot.
        NOTE: apparently this isn't enough,
              nor does stopJ do anything, so it goes to freedriveMode
              and then exits it, which actually stops ur robots at least.
        """
        # when we make a new process, it inherits signal handling.
        # which means we call this more than once.
        # and that could lead to race conditions.
        # but if we exit immediatealy it's fine
        if getpid() != self.pid:
            return
        print('sending 300 speedjs full of zeros and exiting')
        for i in range(300):
            vel_cmd = np.zeros(self.robot_manager.model.nv)
            self.robot_manager.sendQd(vel_cmd)

        # hopefully this actually stops it
        if not self.args.pinocchio_only:
            self.robot_manager.rtde_control.speedStop(1)
            print("sending a stopj as well")
            self.robot_manager.rtde_control.stopJ(1)
            print("putting it to freedrive for good measure too")
            self.robot_manager.rtde_control.freedriveMode()

        if self.args.save_log:
            print("saving log")
            # this does not get run if you exited with ctrl-c
            self.robot_manager.log_manager.storeControlLoopRun(self.log_dict, self.controlLoop.func.__name__, self.final_iteration)
            self.robot_manager.log_manager.saveLog()

        if self.args.plotter:
            self.plotter_manager.terminateProcess()

        if self.args.visualizer:
            self.robot_manager.visualizer_manager.terminateProcess()
        
        # TODO: this obviously only makes sense if you're on ur robots
        # so this functionality should be wrapped in robotmanager
        if not self.args.pinocchio_only:
            self.robot_manager.rtde_control.endFreedriveMode()

        exit()

class RobotManager:
    """
    RobotManager:
    ---------------
    - design goal: rely on pinocchio as much as possible while
                   concealing obvious bookkeeping
    - right now it is assumed you're running this on UR5e so some
      magic numbers are just put to it.
      this will be extended once there's a need for it.
    - at this stage it's just a boilerplate reduction class
      but the idea is to inherit it for more complicated things
      with many steps, like dmp.
      or just showe additional things in, this is python after all
    - you write your controller separately,
      and then drop it into this - there is a wrapper function you put
      around the control loop which handles timing so that you
      actually run at 500Hz and not more.
    - this is probably not the most new-user friendly solution,
      but it's designed for fastest idea to implementation rate.
    - if this was a real programming language, all of these would really be private variables.
      as it currently stands, "private" functions have the '_' prefix 
      while the public getters don't have a prefix.
    - TODO: write out default arguments needed here as well
    """

    # just pass all of the arguments here and store them as is
    # so as to minimize the amount of lines.
    # might be changed later if that seems more appropriate
    def __init__(self, args):
        self.args = args
        self.pinocchio_only = args.pinocchio_only
        if self.args.simulation:
            self.args.robot_ip = "127.0.0.1"
        # load model
        # collision and visual models are none if args.visualize == False
        self.robot_name = args.robot
        if self.robot_name == "ur5e":
            self.model, self.collision_model, self.visual_model, self.data = \
                 get_model()
        if self.robot_name == "heron":
            self.model, self.collision_model, self.visual_model, self.data = \
                 heron_approximation()
        if self.robot_name == "heronros":
            self.model, self.collision_model, self.visual_model, self.data = \
                 heron_approximation()
        if self.robot_name == "mirros":
            self.model, self.collision_model, self.visual_model, self.data = \
                 mir_approximation()
            #self.publisher_vel_base = create_publisher(msg.Twist, '/cmd_vel', 5)
            #self.publisher_vel_base = publisher_vel_base
        if self.robot_name == "gripperlessur5e":
            self.model, self.collision_model, self.visual_model, self.data = \
                 getGripperlessUR5e()
        if self.robot_name == "yumi":
            self.model, self.collision_model, self.visual_model, self.data = \
                 get_yumi_model()

        # create log manager if we're saving logs
        if args.save_log:
            self.log_manager = LogManager(args)
        
        # ur specific magic numbers 
        # NOTE: all of this is ur-specific, and needs to be if-ed if other robots are added.
        # TODO: this is 8 in pinocchio and that's what you actually use 
        # if we're being real lmao
        # the TODO here is make this consistent obviously
        self.n_arm_joints = 6
        # last joint because pinocchio adds base frame as 0th joint.
        # and since this is unintuitive, we add the other variable too
        # so that the control designer doesn't need to think about such bs
        #self.JOINT_ID = 6
        self.ee_frame_id = self.model.getFrameId("tool0")
        if self.robot_name == "yumi":
            self.r_ee_frame_id = self.model.getFrameId("robr_joint_7")
            self.l_ee_frame_id = self.model.getFrameId("robl_joint_7")
            # JUST FOR TEST
            # NOTE 
            # NOTE 
            # NOTE 
            self.ee_frame_id = self.model.getFrameId("robr_joint_7")
        #self.ee_frame_id = self.model.getFrameId("hande_right_finger_joint")
        # TODO: add -1 option here meaning as fast as possible
        self.update_rate = args.ctrl_freq #Hz
        self.dt = 1 / self.update_rate
        # you better not give me crazy stuff
        # and i'm not clipping it, you're fixing it
        self.MAX_ACCELERATION = 1.7
        assert args.acceleration <= self.MAX_ACCELERATION and args.acceleration > 0.0
        # this is the number passed to speedj
        self.acceleration = args.acceleration
        # NOTE: this is evil and everything only works if it's set to 1
        # you really should control the acceleration via the acceleration argument.
        assert args.speed_slider <= 1.0 and args.acceleration > 0.0
        # TODO: these are almost certainly higher
        # NOTE and TODO: speed slider is evil, put it to 1, handle the rest yourself.
        # NOTE: i have no idea what's the relationship between max_qdd and speed slider
        #self.max_qdd = 1.7 * args.speed_slider
        # NOTE: this is an additional kinda evil speed limitation (by this code, not UR).
        # we're clipping joint velocities with this.
        # if your controllers are not what you expect, you might be commanding a very high velocity,
        # which is clipped, resulting in unexpected movement.
        self.MAX_QD = 3.14
        assert args.max_qd <= self.MAX_QD and args.max_qd > 0.0
        self.max_qd = args.max_qd
        self.u_ref_w = np.zeros((1, 6))
        self.gripper = None
        if (self.args.gripper != "none") and not self.pinocchio_only:
            if self.args.gripper == "robotiq":
                self.gripper = RobotiqGripper()
                self.gripper.connect(args.robot_ip, 63352)
                self.gripper.activate()
            if self.args.gripper == "onrobot":
                self.gripper = TWOFG()

        # TODO: specialize for each robot,
        # add reasonable home positions
        self.q = pin.randomConfiguration(self.model, 
                                         -1 * np.ones(self.model.nq),
                                         np.ones(self.model.nq))
        if self.robot_name == "ur5e":
            self.q[-1] = 0.0
            self.q[-2] = 0.0
        # v_q is the generalization of qd for every type of joint.
        # for revolute joints it's qd, but for ex. the planar joint it's the body velocity.
        self.v_q = np.zeros(self.model.nv)
        # same note as v_q, but it's a_q. 
        self.a_q = np.zeros(self.model.nv)

        # start visualize manipulator process if selected.
        # has to be started here because it lives throughout the whole run
        if args.visualizer:
            side_function = partial(manipulatorVisualizer, self.model, self.collision_model, self.visual_model)
            self.visualizer_manager = ProcessManager(args, side_function, {"q" : self.q.copy()}, 0)
            if args.visualize_collision_approximation:
                pass
            # TODO: import the ellipses here, and write an update ellipse function
            # then also call that in controlloopmanager

        # initialize and connect the interfaces
        if not args.pinocchio_only:
            # NOTE: you can't connect twice, so you can't have more than one RobotManager.
            # if this produces errors like "already in use", and it's not already in use,
            # try just running your new program again. it could be that the socket wasn't given
            # back to the os even though you've shut off the previous program.
            print("CONNECTING TO UR5e!")
            self.rtde_control = RTDEControlInterface(args.robot_ip)
            self.rtde_receive = RTDEReceiveInterface(args.robot_ip)
            self.rtde_io = RTDEIOInterface(args.robot_ip)
            self.rtde_io.setSpeedSlider(args.speed_slider)
            # NOTE: the force/torque sensor just has large offsets for no reason,
            # and you need to minus them to have usable readings.
            # we provide this with calibrateFT
            self.wrench_offset = self.calibrateFT()
        else:
            self.wrench_offset = np.zeros(6)

        self.speed_slider = args.speed_slider

        if args.pinocchio_only and args.start_from_current_pose:
            self.rtde_receive = RTDEReceiveInterface(args.robot_ip)
            q = self.rtde_receive.getActualQ()
            q.append(0.0)
            q.append(0.0)
            q = np.array(q)
            self.q = q
            if args.visualizer:
                self.visualizer_manager.sendCommand({"q" : q})


        # do it once to get T_w_e
        self._step()

#######################################################################
#               getters which assume you called step()                #
#######################################################################
    
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
                pin.forwardKinematics(self.model, self.data, q_given, 
                                      np.zeros(self.model.nv), np.zeros(self.model.nv))
                # NOTE: this also returns the frame, so less copying possible
                #pin.updateFramePlacements(self.model, self.data)
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
                pin.forwardKinematics(self.model, self.data, q_given, 
                                      np.zeros(self.model.nv), np.zeros(self.model.nv))
                # NOTE: this also returns the frame, so less copying possible
                #pin.updateFramePlacements(self.model, self.data)
                pin.updateFramePlacement(self.model, self.data, self.r_ee_frame_id)
                pin.updateFramePlacement(self.model, self.data, self.l_ee_frame_id)
                return self.data.oMf[self.l_ee_frame_id].copy(), self.data.oMf[self.r_ee_frame_id].copy()


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
        #self._getWrench()
        # computeAllTerms is certainly not necessary btw
        # but if it runs on time, does it matter? it makes everything available...
        # (includes forward kinematics, all jacobians, all dynamics terms, energies)
        #pin.computeAllTerms(self.model, self.data, self.q, self.v_q)
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
                #self.gripper_past_pos = self.gripper_pos
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
                    #self.gripper_past_pos = self.gripper_pos
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
            #self.T_w_e = self.data.oMf[self.l_ee_frame_id].copy()

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
                #self.gripper_vel = (gripper.get_current_position() - self.gripper_pos) / self.dt
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
            self.wrench = np.array(self.rtde_receive.getActualTCPForce()) - self.wrench_offset
        else:
            # TODO: do something better here (at least a better distribution)
            self.wrench = np.random.random(self.n_arm_joints)


    def _getWrenchInEE(self, step_called=False):
        if self.robot_name != "yumi":
            if not self.pinocchio_only:
                self.wrench = np.array(self.rtde_receive.getActualTCPForce()) - self.wrench_offset
            else:
                # TODO: do something better here (at least a better distribution)
                self.wrench = np.random.random(self.n_arm_joints)
            if not step_called:
                self._getT_w_e()
            # NOTE: this mapping is equivalent to having a purely rotational action 
            # this is more transparent tho
            mapping = np.zeros((6,6))
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
            qd_cmd = np.clip(qd, -1 * self.model.velocityLimit, self.model.velocityLimit)
            #qd[1] = 0
            self.v_q = qd_cmd
            self.q = pin.integrate(self.model, self.q, qd_cmd * self.dt)

        if self.robot_name == "heronros":
            # y-direction is not possible on heron
            qd[1] = 0
            cmd_msg = msg.Twist()
            cmd_msg.linear.x = qd[0]
            cmd_msg.angular.z = qd[2]
            #print("about to publish", cmd_msg)
            self.publisher_vel_base.publish(cmd_msg)
            # good to keep because updating is slow otherwise
            # it's not correct, but it's more correct than not updating
            #self.q = pin.integrate(self.model, self.q, qd * self.dt)

        if self.robot_name == "mirros":
            # y-direction is not possible on heron
            qd[1] = 0
            cmd_msg = msg.Twist()
            cmd_msg.linear.x = qd[0]
            cmd_msg.angular.z = qd[2]
            #print("about to publish", cmd_msg)
            self.publisher_vel_base.publish(cmd_msg)
            # good to keep because updating is slow otherwise
            # it's not correct, but it's more correct than not updating
            #self.q = pin.integrate(self.model, self.q, qd * self.dt)

        if self.robot_name == "gripperlessur5e":
            qd_cmd = np.clip(qd, -1 * self.max_qd, self.max_qd)
            if not self.pinocchio_only:
                self.rtde_control.speedJ(qd_cmd, self.acceleration, self.dt)
            else:
                self.v_q = qd_cmd
                self.q = pin.integrate(self.model, self.q, qd_cmd * self.dt)

        if self.robot_name == "yumi":
            qd_cmd = np.clip(qd, -0.01 * self.model.velocityLimit, 0.01 *self.model.velocityLimit)
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
                print("you didn't select a gripper (no gripper is the default parameter) so no gripping for you")
            return
        if (not self.args.simulation) and (not self.args.pinocchio_only):
            self.gripper.open()
        else:
            print("not implemented yet, so nothing is going to happen!")

    def closeGripper(self):
        if self.gripper is None:
            if self.args.debug_prints:
                print("you didn't select a gripper (no gripper is the default parameter) so no gripping for you")
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
            print("In the following, first 3 numbers are x,y,z position, and second 3 are r,p,y angles")
            print("Here's where the robot is currently. Ensure you know what the base frame is first.")
            print("base frame end-effector pose from pinocchio:\n", \
                    *self.data.oMi[6].translation.round(4), *pin.rpy.matrixToRpy(self.data.oMi[6].rotation).round(4))
            print("UR5e TCP:", *np.array(self.rtde_receive.getActualTCPPose()).round(4))
        # remain with the current orientation
        # TODO: add something, probably rpy for orientation because it's the least number
        # of numbers you need to type in
        Mgoal = T_w_e.copy()
        # this is a reasonable way to do it too, maybe implement it later
        #Mgoal.translation = Mgoal.translation + np.array([0.0, 0.0, -0.1])
        # do a while loop until this is parsed correctly
        while True:
            goal = input("Please enter the target end-effector position in the x.x,y.y,z.z format: ")
            try:
                e = "ok"
                goal_list = goal.split(',')
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
        if self.args.visualizer:
            # TODO document this somewhere
            self.visualizer_manager.sendCommand(
                    {"Mgoal" : Mgoal})
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

    def updateViz(self, viz_dict : dict):
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



class ProcessManager:
    """
    ProcessManager
    --------------
    A way to do processing in a thread (process because GIL) different 
    from the main one which is reserved 
    for ControlLoopManager.
    The primary goal is to process visual input
    from the camera without slowing down control.
    TODO: once finished, map real-time-plotting and 
    visualization with this (it already exists, just not in a manager).
    What this should do is hide the code for starting a process,
    and to enforce the communication defined by the user.
    To make life simpler, all communication is done with Queues.
    There are two Queues - one for commands,
    and the other for receiving the process' output.
    Do note that this is obviously not the silver bullet for every
    possible inter-process communication scenario,
    but the aim of this library is to be as simple as possible,
    not as most performant as possible.
    NOTE: the maximum number of things in the command queue is arbitrarily
        set to 5. if you need to get the result immediately,
        calculate whatever you need in main, not in a side process.
        this is meant to be used for operations that take a long time
        and aren't critical, like reading for a camera.
    """
    # NOTE: theoretically we could pass existing queues so that we don't 
    # need to create new ones, but i won't optimize in advance
    def __init__(self, args, side_function, init_command, comm_direction, init_value=None):
        self.args = args
        self.comm_direction = comm_direction

        # send command to slave process
        if comm_direction == 0:
            self.command_queue = Queue()
            self.side_process = Process(target=side_function, 
                                                     args=(args, init_command, self.command_queue,))
        # get data from slave process
        if comm_direction == 1:
            self.data_queue = Queue()
            self.side_process = Process(target=side_function, 
                                                     args=(args, init_command, self.data_queue,))
        # share data in both directions via shared memory with 2 buffers
        # - one buffer for master to slave
        # - one buffer for slave to master
        if comm_direction == 2:
            self.command_queue = Queue()
            self.data_queue = Queue()
            self.side_process = Process(target=side_function, 
                                                     args=(args, init_command, self.command_queue, self.data_queue,))
        # shared memory both ways
        # one lock because i'm lazy
        # but also, it's just copy-pasting 
        # we're in python, and NOW do we get picky with copy-pasting???????
        if comm_direction == 3:
            # "sending" the command via shared memory
            # TODO: the name should be random and send over as function argument
            shm_name = "command"
            self.shm_cmd = shared_memory.SharedMemory(shm_name, create=True, size=init_command.nbytes)
            self.shared_command = np.ndarray(init_command.shape, dtype=init_command.dtype, buffer=self.shm_cmd.buf)
            self.shared_command[:] = init_command[:]
            # same lock for both
            self.shared_command_lock = Lock()
            # getting data via different shared memory
            shm_data_name = "data"
            # size is chosen arbitrarily but 10k should be more than enough for anything really
            self.shm_data = shared_memory.SharedMemory(shm_data_name, create=True, size=10000)
            # initialize empty
            p = pickle.dumps(None)
            self.shm_data.buf[:len(p)] = p
            # the process has to create its shared memory
            self.side_process = Process(target=side_function, 
                                         args=(args, init_command, shm_name, self.shared_command_lock, self.shm_data,))
        # networking client (server can use comm_direction 0)
        if comm_direction == 4:
            from ur_simple_control.networking.util import DictPb2EncoderDecoder
            self.encoder_decoder = DictPb2EncoderDecoder()
            self.msg_code = self.encoder_decoder.dictToMsgCode(init_command)
            # TODO: the name should be random and send over as function argument
            shm_name = "client_socket" + str(np.random.randint(0, 1000))
            # NOTE: size is max size of the recv buffer too,
            # and the everything blows up if you manage to fill it atm
            self.shm_msg = shared_memory.SharedMemory(shm_name, create=True, size=1024)
            # need to initialize shared memory with init value
            # NOTE: EVIL STUFF SO PICKLING ,READ NOTES IN networking/client.py
            #init_val_as_msg = self.encoder_decoder.dictToSerializedPb2Msg(init_value)
            #self.shm_msg.buf[:len(init_val_as_msg)] = init_val_as_msg
            pickled_init_value = pickle.dumps(init_value)
            self.shm_msg.buf[:len(pickled_init_value)] = pickled_init_value
            self.lock = Lock()
            self.side_process = Process(target=side_function, 
                                         args=(args, init_command, shm_name, self.lock))
        if type(side_function) == partial:
            self.side_process.name = side_function.func.__name__
        else:
            self.side_process.name = side_function.__name__ + "_process"
        self.latest_data = init_value

        self.side_process.start()
        if self.args.debug_prints:
            print(f"PROCESS_MANAGER: i am starting {self.side_process.name}")


    # TODO: enforce that
    # the key should be a string containing the command,
    # and the value should be the data associated with the command,
    # just to have some consistency
    def sendCommand(self, command : typing.Union[dict, np.ndarray]):
        """
        sendCommand
        ------------
        assumes you're calling from controlLoop and that
        you want a non-blocking call.
        the maximum number of things in the command queue is arbitrarily
        set to 5. if you need to get the result immediately,
        calculate whatever you need in main, not in a side process.

        if comm_direction == 3:
        sendCommandViaSharedMemory
        ---------------------------
        instead of having a queue for the commands, have a shared memory variable.
        this makes sense if you want to send the latest command only,
        instead of stacking up old commands in a queue.
        the locking and unlocking of the shared memory happens here 
        and you don't have to think about it in the control loop nor
        do you need to pollute half of robotmanager or whatever else
        to deal with this.
        """
        if self.comm_direction != 3:
            if self.command_queue.qsize() < 5:
                self.command_queue.put_nowait(command)

        if self.comm_direction == 3:
            assert type(command) == np.ndarray
            assert command.shape == self.shared_command.shape
            self.shared_command_lock.acquire()
            self.shared_command[:] = command[:]
            self.shared_command_lock.release()


    def getData(self):
        if self.comm_direction < 3:
            if not self.data_queue.empty():
                self.latest_data = self.data_queue.get_nowait()
        if self.comm_direction == 3:
            self.shared_command_lock.acquire()
            # here we should only copy, release the lock, then deserialize
            self.latest_data = pickle.loads(self.shm_data.buf)
            self.shared_command_lock.release()
        if self.comm_direction == 4:
            self.lock.acquire()
            #data_copy = copy.deepcopy(self.shm_msg.buf)
            # REFUSES TO WORK IF YOU DON'T PRE-CROP HERE!!!
            # MAKES ABSOLUTELY NO SENSE!!! READ MORE IN ur_simple_control/networking/client.py
            # so we're decoding there, pickling, and now unpickling.
            # yes, it's incredibly stupid
            #new_data = self.encoder_decoder.serializedPb2MsgToDict(self.shm_msg.buf, self.msg_code)
            new_data = pickle.loads(self.shm_msg.buf)
            self.lock.release()
            if len(new_data) > 0:
                self.latest_data = new_data
            #print("new_data", new_data)
            #print("self.latest_data", self.latest_data)
            #self.latest_data = self.encoder_decoder.serializedPb2MsgToDict(data_copy, self.msg_code)
        return copy.deepcopy(self.latest_data)

    def terminateProcess(self):
        if self.comm_direction == 3:
            self.shm_cmd.close()
            self.shm_cmd.unlink()
        if (self.comm_direction != 3) and (self.comm_direction != 1):
            if self.args.debug_prints:
                print(f"i am putting befree in {self.side_process.name}'s command queue to stop it")
            self.command_queue.put_nowait("befree")
        try:
            self.side_process.terminate()
            if self.args.debug_prints:
                print(f"terminated {self.side_process.name}")
        except AttributeError:
            if self.args.debug_prints:
                print(f"{self.side_process.name} is dead already")


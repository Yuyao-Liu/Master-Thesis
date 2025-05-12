from argparse import Namespace
from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.robots.interfaces.dual_arm_interface import DualArmInterface
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface
from smc.multiprocessing.process_manager import ProcessManager
from smc.visualization.plotters import realTimePlotter

from functools import partial
import signal
import time
import numpy as np
from collections import deque
import copy
from os import getpid


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

    def __init__(
        self,
        robot_manager: AbstractRobotManager,
        controlLoop: partial,
        args: Namespace,
        save_past_item: dict[str, np.ndarray],
        log_item: dict[str, np.ndarray],
    ):
        signal.signal(signal.SIGINT, self.stopHandler)
        self.pid = getpid()
        self.max_iterations: int = args.max_iterations
        self.robot_manager: AbstractRobotManager = robot_manager
        self.controlLoop = controlLoop  # TODO: declare partial function type
        self.final_iteration = -1  # because we didn't even start yet
        self.args: Namespace = args
        self.iter_n: int = 0
        self.past_data: dict[str, deque[np.ndarray]] = {}
        self.current_iteration = 0
        # NOTE: viz update rate is a magic number that seems to work fine and i don't have
        # any plans to make it smarter
        if args.viz_update_rate < 0 and args.ctrl_freq > 0:
            self.viz_update_rate: int = int(np.ceil(self.args.ctrl_freq / 25))
        else:
            self.viz_update_rate: int = args.viz_update_rate
        # save_past_dict has to have the key and 1 example of what you're saving
        # so that it's type can be inferred (but we're in python so types don't really work).
        # the good thing is this way you also immediatelly put in the initial values
        for key in save_past_item:
            self.past_data[key] = deque()
            # immediatelly populate every deque with initial values
            for _ in range(self.args.past_window_size):
                # deepcopy just in case, better safe than sorry plus it's during initialization,
                # not real time
                self.past_data[key].append(copy.deepcopy(save_past_item[key]))

        # similar story for log_dict as for past_data,
        # except this is not used in the control loop,
        # we don't predeclare sizes, but instead
        # just shove items into linked lists (python lists) in dictionaries (hash-maps)
        self.log_dict: dict[str, list[np.ndarray]] = {}
        for key in log_item:
            self.log_dict[key] = []

        if self.args.plotter:
            self.plotter_manager: ProcessManager = ProcessManager(
                args, realTimePlotter, log_item, 0
            )

    def run_one_iter(self, i):
        """
        run
        ---
        do timing to run at 500Hz.
        also handle the number of iterations.
        it's the controlLoop's responsibility to break if it achieved it's goals.
        this is done via the breakFlag
        """
        self.current_iteration += 1
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
        # if len(self.log_dict) > 0:
        for key in log_item:
            # if key not in self.log_dict.keys():
            #    raise KeyError("you need to provide log items you promised!")
            #    break
            self.log_dict[key].append(log_item[key])

        # TODO: do it this way if running on the real robot.
        # but if not, we want to control the speed of the simulation,
        # and how much we're plotting.
        # so this should be an argument that is use ONLY if we're in simulation
        if i % self.viz_update_rate == 0:
            # don't send what wasn't ready
            if self.args.visualizer:
                self.robot_manager.visualizer_manager.sendCommand(
                    {
                        "q": self.robot_manager._q,
                    }
                )
                # NOTE: for dual armed robots it displays T_w_abs
                if issubclass(self.robot_manager.__class__, SingleArmInterface):
                    self.robot_manager.visualizer_manager.sendCommand(
                        {
                            "T_w_e": self.robot_manager.T_w_e,
                        }
                    )
                # TODO: implement in visualizer
                #                if issubclass(self.robot_manager.__class__, DualArmInterface):
                #                    self.robot_manager.visualizer_manager.sendCommand(
                #                        {
                #                            "T_w_l": self.robot_manager.T_w_l,
                #                            "T_w_r": self.robot_manager.T_w_r,
                #                        }
                #                    )
                if issubclass(self.robot_manager.__class__, MobileBaseInterface):
                    self.robot_manager.visualizer_manager.sendCommand(
                        {"T_base": self.robot_manager.T_w_b}
                    )

                if self.args.visualize_collision_approximation:
                    raise NotImplementedError
                # TODO: here call robot manager's update ellipses function or whatever approximation is used

            if self.args.plotter:
                # don't put new stuff in if it didn't handle the previous stuff.
                # it's a plotter, who cares if it's late.
                # the number 5 is arbitrary
                self.plotter_manager.sendCommand(log_item)
        return breakFlag

    def run(self):
        self.final_iteration = 0
        self.current_iteration = 0
        for i in range(self.max_iterations):
            self.current_iteration = i
            start = time.time()
            breakFlag = self.run_one_iter(i)

            # break if done
            if breakFlag:
                break

            # sleep for the rest of the frequency cycle
            end = time.time()
            diff = end - start
            if (self.robot_manager.dt < diff) and (self.args.ctrl_freq > 0):
                if self.args.debug_prints:
                    print("missed deadline by", diff - self.robot_manager.dt)
                continue
            else:
                if self.args.real or self.args.ctrl_freq > 0:
                    time.sleep(self.robot_manager.dt - diff)

        ######################################################################
        # for over
        ######################################################################
        if self.args.plotter:
            self.plotter_manager.terminateProcess()
        if self.args.save_log:
            self.robot_manager._log_manager.storeControlLoopRun(
                self.log_dict, self.controlLoop.func.__name__, self.final_iteration
            )
        if i < self.max_iterations - 1:
            if self.args.debug_prints:
                print("success in", i, "iterations!")
        else:
            print("FAIL: did not succed in", self.max_iterations, "iterations")
            # self.stopHandler(None, None)

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
        # TODO: you want this kinda bullshit to be defined on a per-robot basis,
        # and put it into robot_manager.stopRobot()
        print("sending 300 speedjs full of zeros and exiting")
        for _ in range(300):
            vel_cmd = np.zeros(self.robot_manager.nv)
            self.robot_manager.sendVelocityCommand(vel_cmd)

        self.robot_manager.stopRobot()

        if self.args.save_log:
            print("saving log")
            # this does not get run if you exited with ctrl-c
            self.robot_manager._log_manager.storeControlLoopRun(
                self.log_dict, self.controlLoop.func.__name__, self.final_iteration
            )
            self.robot_manager._log_manager.saveLog()

        if self.args.plotter:
            self.plotter_manager.terminateProcess()

        if self.args.visualizer:
            self.robot_manager.visualizer_manager.terminateProcess()

        exit()

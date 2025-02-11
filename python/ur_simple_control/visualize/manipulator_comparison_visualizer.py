import numpy as np
import argparse
import os
import time
from itertools import zip_longest
from multiprocessing import Process, Queue
from ur_simple_control.managers import getMinimalArgParser, RobotManager, ProcessManager
from ur_simple_control.util.logging_utils import LogManager
from ur_simple_control.visualize.visualize import manipulatorComparisonVisualizer, logPlotter
from functools import partial

def getLogComparisonArgs():
    parser = getMinimalArgParser()
#    parser = getClikArgs(parser)
#    parser = getDMPArgs(parser)
#    parser = getBoardCalibrationArgs(parser)
    parser.add_argument('--log-file1', type=str, \
            help="first log file to load for comparison")
    parser.add_argument('--log-file2', type=str, \
            help="first log file to load for comparison")
    args = parser.parse_args()
    # these are obligatory
    args.visualize_manipulator = False
    args.pinocchio_only = True
    args.simulation = False
    return args

# TODO: use ProcessManager for this
class ManipulatorComparisonManager:
    def __init__(self, args):
        self.args = args
        # these are obligatory
        args.visualize_manipulator = False
        args.pinocchio_only = True
        args.simulation = False

        self.robot1 = RobotManager(args)
        self.robot2 = RobotManager(args)

        # no two loops will have the same amount of timesteps.
        # we need to store the last available step for both robots.
        self.lastq1 = np.zeros(self.robot1.model.nq)
        self.lastq2 = np.zeros(self.robot2.model.nq)

        if os.path.exists(args.log_file1):
            self.logm1 = LogManager(None)
            self.logm1.loadLog(args.log_file1)
        else:
            print("you did not give me a valid path for log1, exiting")
            exit()
        if os.path.exists(args.log_file2):
            self.logm2 = LogManager(None)
            self.logm2.loadLog(args.log_file2)
        else:
            print("you did not give me a valid path for log2, exiting")
            exit()

        side_function = partial(manipulatorComparisonVisualizer, self.robot1.model, 
                                self.robot1.visual_model, self.robot1.collision_model)
        cmd = (np.zeros(self.robot1.model.nq), np.ones(self.robot2.model.nq))
        self.visualizer_manager = ProcessManager(args, side_function,
                                                 cmd, 2)
        # wait until it's ready (otherwise you miss half the sim potentially)
        # 5 seconds should be more than enough,
        # and i don't want to complicate this code by complicating IPC
        time.sleep(5)
        self.visualizer_manager.getData()

        ###########################################
        #  in case you will want log plotters too  #
        ###########################################
        self.log_plotters = []


    # NOTE i assume what you want to plot is a time-indexed with
    # the same frequency that the control loops run in.
    # if it's not then it's pointless to have a running plot anyway.
    # TODO: put these in a list so that we can have multiple plots at the same time
    class RunningPlotter:
        def __init__(self, args, log, log_plotter_time_start, log_plotter_time_stop):
            self.time_start = log_plotter_time_start
            self.time_stop = log_plotter_time_stop
            side_function = partial(logPlotter, log)
            self.plotter_manager = ProcessManager(args, side_function, None, 2)
            self.plotter_manager.getData()

    def createRunningPlot(self, log, log_plotter_time_start, log_plotter_time_stop):
        self.log_plotters.append(self.RunningPlotter(self.args, log, log_plotter_time_start, log_plotter_time_stop))

    def updateViz(self, q1, q2, time_index):
        self.visualizer_manager.sendCommand((q1, q2))
        for log_plotter in self.log_plotters:
            if (time_index >= log_plotter.time_start) and (time_index < log_plotter.time_stop):
                log_plotter.plotter_manager.sendCommand(time_index - log_plotter.time_start)
                log_plotter.plotter_manager.getData()
        self.visualizer_manager.getData()

    # NOTE: this uses slightly fancy python to make it bareable to code
    # NOTE: dict keys are GUARANTEED to be in insert order from python 3.7 onward
    def visualizeWholeRuns(self):
        time_index = 0 
        for control_loop1, control_loop2 in zip_longest(self.logm1.loop_logs, self.logm2.loop_logs):
            print(f'run {self.logm1.args.run_name}, controller: {control_loop1}')
            print(f'run {self.logm2.args.run_name}, controller: {control_loop2}')
            # relying on python's default thing.toBool()
            if not control_loop1:
                print(f"run {self.logm1.args.run_name} is finished")
                q1 = self.lastq1
                for q2 in self.logm2.loop_logs[control_loop2]['qs']:
                    self.updateViz(q1, q2, time_index)
                    time_index += 1
                print(f"run {self.logm2.args.run_name} is finished")
            if not control_loop2:
                print(f"run {self.logm2.args.run_name} is finished")
                q2 = self.lastq2
                for q1 in self.logm1.loop_logs[control_loop1]['qs']:
                    self.updateViz(q1, q2, time_index)
                    time_index += 1
                print(f"run {self.logm1.args.run_name} is finished")
            if control_loop1 and control_loop2:
                for q1, q2 in zip_longest(self.logm1.loop_logs[control_loop1]['qs'], \
                            self.logm2.loop_logs[control_loop2]['qs']):
                    if not (q1 is None):
                        self.lastq1 = q1
                    if not (q2 is None):
                        self.lastq2 = q2
                    self.updateViz(self.lastq1, self.lastq2, time_index)
                    time_index += 1



# TODO: update
if __name__ == "__main__":
    args = getLogComparisonArgs()
    cmp_manager = ManipulatorComparisonManager(args)
    log_plot = {'random_noise' : np.random.normal(size=(1000, 2))}
    cmp_manager.createRunningPlot(log_plot, 200, 1200)
    cmp_manager.visualizeWholeRuns()
    time.sleep(100)
    cmp_manager.manipulator_visualizer_cmd_queue.put("befree")
    print("main done")
    time.sleep(0.1)
    cmp_manager.manipulator_visualizer_process.terminate()
    if args.debug_prints:
        print("terminated manipulator_visualizer_process")

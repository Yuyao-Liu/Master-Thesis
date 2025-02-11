# PYTHON_ARGCOMPLETE_OK
import numpy as np
import time
import pinocchio as pin
import argcomplete, argparse
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
from ur_simple_control.clik.clik import getClikArgs, getClikController, controlLoopClik, moveL, compliantMoveL


def get_args():
    parser = getMinimalArgParser()
    parser.description = 'Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!'
    parser = getClikArgs(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = get_args()
    robot = RobotManager(args)
    print(robot.getT_w_e())
    Mgoal = robot.defineGoalPointCLI()
    compliantMoveL(args, robot, Mgoal)
    #moveL(args, robot, Mgoal)
    robot.closeGripper()
    robot.openGripper()
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()
    #loop_manager.stopHandler(None, None)


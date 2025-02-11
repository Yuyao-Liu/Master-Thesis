import pinocchio as pin
import numpy as np
import time
import argparse
from functools import partial
from ur_simple_control.managers import (
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
)
from ur_simple_control.basics.basics import freedriveUntilKeyboard
import pickle


def get_args():
    parser = getMinimalArgParser()
    parser.description = (
        "freedrive with keyboard bindings to save poses and joint angles"
    )
    # add more arguments here from different Simple Manipulator Control modules
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    robot = RobotManager(args)

    pose_n_q_dict = freedriveUntilKeyboard(args, robot)
    file = open("./pose_n_q_dict.pickle", "wb")
    pickle.dump(pose_n_q_dict, file)
    file.close()
    print(pose_n_q_dict)

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.saveLog()
    # loop_manager.stopHandler(None, None)

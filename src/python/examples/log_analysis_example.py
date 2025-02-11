import pinocchio as pin
import numpy as np
import argparse
from ur_simple_control.util.logging_utils import LogManager
from ur_simple_control.visualize.manipulator_comparison_visualizer import getLogComparisonArgs, ManipulatorComparisonManager


if __name__ == "__main__": 
    args = getLogComparisonArgs()
    log_manager = LogManager(None)
    log_manager.loadLog(args.log_file1)
#    mcm = ManipulatorComparisonManager(args)
#    mcm.visualizeWholeRuns()
    # TODO: put this into tabs
    log_manager.plotAllControlLoops()


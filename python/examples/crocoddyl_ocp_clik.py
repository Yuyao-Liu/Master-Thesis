# PYTHON_ARGCOMPLETE_OK
import numpy as np
import time
import argparse
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
from ur_simple_control.optimal_control.crocoddyl_optimal_control import createCrocoIKOCP, solveCrocoOCP
from ur_simple_control.optimal_control.get_ocp_args import get_OCP_args
from ur_simple_control.basics.basics import followKinematicJointTrajP
from ur_simple_control.util.logging_utils import LogManager
from ur_simple_control.visualize.visualize import plotFromDict
import pinocchio as pin
import crocoddyl
import argcomplete


def get_args():
    parser = getMinimalArgParser()
    parser = get_OCP_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    args = get_args()
    robot = RobotManager(args)
    # TODO: put this back for nicer demos
    #Mgoal = robot.defineGoalPointCLI()
    Mgoal = pin.SE3.Random()

    if args.visualize_manipulator:
        # TODO document this somewhere
        robot.visualizer_manager.sendCommand({"Mgoal" : Mgoal})

    # create and solve the optimal control problem of
    # getting from current to goal end-effector position.
    # reference is position and velocity reference (as a dictionary),
    # while solver is a crocoddyl object containing a lot more information
    # starting state
    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    problem = createCrocoIKOCP(args, robot, x0, Mgoal)
    # this shouldn't really depend on x0 but i can't be bothered
    reference, solver = solveCrocoOCP(args, robot, problem, x0)

# NOTE: IF YOU PLOT SOMETHING OTHER THAN REAL-TIME PLOTTING FIRST IT BREAKS EVERYTHING
#    if args.solver == "boxfddp":
#        log = solver.getCallbacks()[1]
#        crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=True)

    # we need a way to follow the reference trajectory,
    # both because there can be disturbances,
    # and because it is sampled at a much lower frequency
    followKinematicJointTrajP(args, robot, reference)

    print("final position:")
    print(robot.getT_w_e())


    if not args.pinocchio_only:
        robot.stopRobot()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()
        robot.log_manager.plotAllControlLoops()


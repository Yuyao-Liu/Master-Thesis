# PYTHON_ARGCOMPLETE_OK
import pinocchio as pin
import numpy as np
import time
import argparse
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
from ur_simple_control.optimal_control.create_pinocchio_casadi_ocp import createCasadiIKObstacleAvoidanceOCP
from ur_simple_control.optimal_control.get_ocp_args import get_OCP_args
from ur_simple_control.basics.basics import followKinematicJointTrajP
import argcomplete

def get_args():
    parser = getMinimalArgParser()
    parser = get_OCP_args(parser)
    parser.description = 'optimal control in pinocchio.casadi for obstacle avoidance'
    # add more arguments here from different Simple Manipulator Control modules
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = get_args()
    robot = RobotManager(args)
    #T_goal = robot.defineGoalPointCLI()
    #T_goal = pin.SE3.Random() 
    T_goal = robot.defineGoalPointCLI()
    T_goal.rotation = robot.getT_w_e().rotation
    if args.visualize_manipulator:
        robot.updateViz({"Mgoal" : T_goal})
    reference, opti = createCasadiIKObstacleAvoidanceOCP(args, robot, T_goal)
    followKinematicJointTrajP(args, robot, reference)
    

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()

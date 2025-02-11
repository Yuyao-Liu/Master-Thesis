# PYTHON_ARGCOMPLETE_OK
import numpy as np
import time
import argparse, argcomplete
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
from ur_simple_control.optimal_control.get_ocp_args import get_OCP_args
from ur_simple_control.optimal_control.crocoddyl_optimal_control import createCrocoEEPathFollowingOCP, solveCrocoOCP
from ur_simple_control.optimal_control.crocoddyl_mpc import CrocoEndEffectorPathFollowingMPCControlLoop, CrocoEndEffectorPathFollowingMPC
from ur_simple_control.basics.basics import followKinematicJointTrajP
from ur_simple_control.util.logging_utils import LogManager
from ur_simple_control.visualize.visualize import plotFromDict
from ur_simple_control.clik.clik import getClikArgs
import pinocchio as pin
import crocoddyl
import importlib.util

def path(T_w_e, i):
    # 2) do T_mobile_base_ee_pos to get 
    # end-effector reference frame(s)

    # generate bullshit just to see it works
    path = []
    t = i * robot.dt
    for i in range(args.n_knots):
        t += 0.01
        new = T_w_e.copy()
        translation = 2 * np.array([np.cos(t/20), np.sin(t/20), 0.3])
        new.translation = translation
        path.append(new)
    return path

def get_args():
    parser = getMinimalArgParser()
    parser = get_OCP_args(parser)
    parser = getClikArgs(parser) # literally just for goal error
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    args = get_args()
    if importlib.util.find_spec('mim_solvers'):
        import mim_solvers
    robot = RobotManager(args)
    #time.sleep(5)

    # create and solve the optimal control problem of
    # getting from current to goal end-effector position.
    # reference is position and velocity reference (as a dictionary),
    # while solver is a crocoddyl object containing a lot more information
    # starting state
    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    robot._step()

    problem = createCrocoEEPathFollowingOCP(args, robot, x0)
    CrocoEndEffectorPathFollowingMPC(args, robot, x0, path)

    print("final position:")
    print(robot.getT_w_e())

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if not args.pinocchio_only:
        robot.stopRobot()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()
    #loop_manager.stopHandler(None, None)


from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc import getRobotFromArgs
from smc.control.cartesian_space import moveL

from argparse import Namespace
import numpy as np
from pinocchio import SE3
import copy
from functools import partial


def clikCartesianPathIntoJointPath(
    ik_solver,
    path: list[SE3],
    q_init: np.ndarray,
    t_final: float,
    args: Namespace,
    robot: AbstractRobotManager,
) -> np.ndarray:
    """
    clikCartesianPathIntoJointPath
    ------------------------------
    functionality
    ------------
    Follows a provided Cartesian path,
    creates a joint space trajectory for it.
    Output format and timing works only for what the dmp code expects
    because it's only used there,
    and I never gave that code a lift-up it needs.

    return
    ------
    - joint_space_trajectory to follow the given path.
    """
    # we don't know how many there will be, so a linked list is
    # clearly the best data structure here (instert is o(1) still,
    # and we aren't pressed on time when turning it into an array later)
    qs = []
    # let's use the functions we already have. to do so
    # we need to create a new RobotManagerAbstract with arguments for simulation,
    # otherwise weird things will happen.
    # we keep all the other args intact
    sim_args = copy.deepcopy(args)
    sim_args.pinocchio_only = True
    sim_args.ctrl_freq = -1
    sim_args.plotter = False
    sim_args.visualizer = False
    sim_args.save_log = False  # we're not using sim robot outside of this
    sim_args.max_iterations = 10000  # more than enough
    if type(ik_solver) is partial:
        sim_args.ik_solver = ik_solver.func.__name__
    else:
        sim_args.ik_solver = ik_solver.__name__
    sim_robot = getRobotFromArgs(sim_args)
    sim_robot._q = q_init.copy()
    sim_robot._step()
    for pose in path:
        moveL(sim_args, sim_robot, pose)
        if args.viz_test_path:
            robot.updateViz(
                {"q": sim_robot.q, "T_w_e": sim_robot.T_w_e, "point": sim_robot.T_w_e}
            )
            # time.sleep(0.005)
        qs.append(sim_robot.q)

    ##############################################
    #  save the obtained joint-space trajectory  #
    ##############################################
    qs = np.array(qs)
    # we're putting a dmp over this so we already have the timing ready
    # TODO: make this general, you don't want to depend on other random
    # arguments (make this one traj_time, then put tau0 = traj_time there
    t = np.linspace(0, t_final, len(qs)).reshape((len(qs), 1))
    joint_trajectory = np.hstack((t, qs))
    # TODO handle saving more consistently/intentionally
    # (although this definitely works right now and isn't bad, just mid)
    # os.makedir -p a data dir and save there, this is ugly
    # TODO: check if we actually need this and if not remove the saving
    # whatever code uses this is responsible to log it if it wants it,
    # let's not have random files around.
    np.savetxt(
        "./parameters/joint_trajectory.csv", joint_trajectory, delimiter=",", fmt="%.5f"
    )
    return joint_trajectory

import pinocchio as pin
import numpy as np
import copy
import argparse
from functools import partial
from ur_simple_control.managers import ControlLoopManager, RobotManager, ProcessManager
import time
from qpsolvers import solve_qp
import argparse
import importlib
import proxsuite

if importlib.util.find_spec("shapely"):
    from ur_simple_control.path_generation.planner import (
        path2D_to_timed_SE3,
        pathPointFromPathParam,
    )


def getClikArgs(parser):
    """
    getClikArgs
    ------------
    Every clik-related magic number, flag and setting is put in here.
    This way the rest of the code is clean.
    Force filtering is considered a core part of these control loops
    because it's not necessarily the same as elsewhere.
    If you want to know what these various arguments do, just grep
    though the code to find them (but replace '-' with '_' in multi-word arguments).
    All the sane defaults are here, and you can change any number of them as an argument
    when running your program. If you want to change a drastic amount of them, and
    not have to run a super long commands, just copy-paste this function and put your
    own parameters as default ones.
    """
    # parser = argparse.ArgumentParser(description='Run closed loop inverse kinematics \
    #        of various kinds. Make sure you know what the goal is before you run!',
    #        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ####################
    #  clik arguments  #
    ####################
    parser.add_argument(
        "--goal-error",
        type=float,
        help="the final position error you are happy with",
        default=1e-2,
    )
    parser.add_argument(
        "--tikhonov-damp",
        type=float,
        help="damping scalar in tikhonov regularization",
        default=1e-3,
    )
    # TODO add the rest
    parser.add_argument(
        "--clik-controller",
        type=str,
        help="select which click algorithm you want",
        default="dampedPseudoinverse",
        choices=[
            "dampedPseudoinverse",
            "jacobianTranspose",
            "invKinmQP",
            "QPproxsuite",
        ],
    )

    ###########################################
    #  force sensing and feedback parameters  #
    ###########################################
    parser.add_argument(
        "--alpha",
        type=float,
        help="force feedback proportional coefficient",
        default=0.01,
    )
    parser.add_argument(
        "--beta", type=float, help="low-pass filter beta parameter", default=0.01
    )

    ###############################
    #  path following parameters  #
    ###############################
    parser.add_argument(
        "--max-init-clik-iterations",
        type=int,
        help="number of max clik iterations to get to the first point",
        default=10000,
    )
    parser.add_argument(
        "--max-running-clik-iterations",
        type=int,
        help="number of max clik iterations between path points",
        default=1000,
    )
    parser.add_argument(
        "--viz-test-path",
        action=argparse.BooleanOptionalAction,
        help="number of max clik iterations between path points",
        default=False,
    )

    return parser


#######################################################################
#                             controllers                             #
#######################################################################
"""
controllers general
-----------------------
really trully just the equation you are running.
ideally, everything else is a placeholder,
meaning you can swap these at will.
if a controller has some additional arguments,
those are put in via functools.partial in getClikController,
so that you don't have to worry about how your controller is handled in
the actual control loop after you've put it in getClikController,
which constructs the controller from an argument to the file.
"""
def compute_manipulability(J):
    # print(np.linalg.det(J @ J.T))
    return np.sqrt(np.linalg.det(J @ J.T) + np.eye(J.shape[0], J.shape[0]) * 1e-6)

def dampedPseudoinverse(tikhonov_damp, J, err_vector):
    # tikhonov_damp = 0
    qd = (
        J.T
        @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * tikhonov_damp)
        @ err_vector
    )
    # print(err_vector)
    return qd

def dPi_adptive_manipulability(tikhonov_damp, J, err_vector):
    """
    Improved version of the damped pseudo-inverse computation with manipulability-based adaptive damping.
    
    Parameters:
    tikhonov_damp (float): The base damping factor used for regularization.
    J (numpy.ndarray): The Jacobian matrix of the system.
    err_vector (numpy.ndarray): The error vector representing the desired task-space velocity.

    Returns:
    numpy.ndarray: The computed joint velocity vector.
    """
    # Compute the manipulability measure of the Jacobian
    manipulability = compute_manipulability(J)
    
    # Increase damping when manipulability is low to prevent numerical instability
    lambda_adaptive = tikhonov_damp / (manipulability + 1e-6)

    # Perform Singular Value Decomposition (SVD) on the Jacobian
    U, S, Vt = np.linalg.svd(J, full_matrices=False)  # full_matrices=False avoids dimension mismatch

    # Construct the filtered singular values for the damped pseudo-inverse
    S_filtered = np.diag([s / (s**2 + lambda_adaptive) if s > 1e-4 else 0 for s in S])

    # Compute the Moore-Penrose pseudo-inverse
    J_pseudo = Vt.T @ S_filtered @ U.T

    # Compute the joint velocity
    qd = J_pseudo @ err_vector

    return qd

def dPi_Weighted(tikhonov_damp, J, err_vector, mode):
    """
    Computes the weighted damped pseudo-inverse of the Jacobian matrix 
    with adaptive damping based on manipulability.

    Parameters:
    tikhonov_damp (float): Base damping factor for regularization.
    J (numpy.ndarray): Jacobian matrix of the system.
    err_vector (numpy.ndarray): Error vector representing the desired task-space velocity.
    mode (str): Operation mode, either "moveL" (moving to a target point) or "moveu" (moving with a given velocity).

    Returns:
    numpy.ndarray: Computed joint velocity vector.
    """

    # Set joint weights based on the mode
    if mode == "moveL":
        # Weights for moving to a target position
        joint_weights = np.array([1, 0.1, 1, 1, 1, 1, 1, 0.1, 1, 1])
    elif mode == "moveu":
        # Weights for moving with a given velocity
        joint_weights = np.array([10, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1])
    else:
        raise ValueError("Invalid mode. Use 'moveL' or 'moveu'")

    # Compute the manipulability measure using only the arm's Jacobian
    J_arm = J[:, 2:]  # Extract the relevant portion of the Jacobian
    manipulability = compute_manipulability(J_arm)

    # Compute the adaptive damping factor
    lambda_adaptive = tikhonov_damp / manipulability

    # Construct the weight matrix
    W = np.diag(joint_weights)

    # Compute the weighted damped pseudo-inverse using the formula:
    # J_pseudo = W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λI)⁻¹
    JWJ = J @ np.linalg.inv(W) @ J.T + lambda_adaptive * np.eye(J.shape[0])
    J_pseudo = np.linalg.inv(W) @ J.T @ np.linalg.inv(JWJ)

    # Compute the joint velocity
    qd = J_pseudo @ err_vector

    return qd

def taskPriorityInverse(tikhonov_damp, J, err_vector, base_dof=2):
    """
    Computes the task-priority inverse kinematics, prioritizing arm joints 
    and only using the base if necessary.

    Parameters:
    tikhonov_damp (float): Damping factor for regularization.
    J (numpy.ndarray): The full Jacobian matrix of the system.
    err_vector (numpy.ndarray): The error vector representing the desired task-space velocity.
    base_dof (int, optional): The number of degrees of freedom (DOF) of the base. Default is 2.

    Returns:
    numpy.ndarray: The computed joint velocity vector.
    """

    # Separate the Jacobian into arm and base components
    J_arm = J[:, base_dof:]   # Arm Jacobian (6 × arm_dof)
    J_base = J[:, :base_dof]  # Base Jacobian (6 × base_dof)
    
    # Step 1: Solve using only the arm joints
    J_arm_pseudo = J_arm.T @ np.linalg.inv(
        J_arm @ J_arm.T + np.eye(J_arm.shape[0], J_arm.shape[0]) * tikhonov_damp
    )  # Compute pseudo-inverse for the arm
    qd_arm = J_arm_pseudo @ err_vector  # Compute arm joint velocities

    # Compute the remaining error that cannot be resolved by the arm alone
    remaining_err = err_vector - J_arm @ qd_arm

    # Step 2: If the remaining error is significant, allow the base to move
    if np.linalg.norm(remaining_err) > 1e-3:
        J_base_pseudo = J_base.T @ np.linalg.inv(
            J_base @ J_base.T + np.eye(J_base.shape[0], J_base.shape[0]) * tikhonov_damp
        )  # Compute pseudo-inverse for the base
        qd_base = J_base_pseudo @ remaining_err  # Compute base movement
    else:
        qd_base = np.zeros(J_base.shape[1])  # Keep the base stationary

    # Combine base and arm velocities into a single joint velocity vector
    qd = np.concatenate((qd_base, qd_arm))
    
    return qd

def dPi_Weighted_nullspace(tikhonov_damp, q, J, err_vector, mode):
    """
    Computes the weighted damped pseudo-inverse with null space motion optimization.

    Parameters:
    tikhonov_damp (float): Base damping factor for regularization.
    q (numpy.ndarray): Current joint positions.
    J (numpy.ndarray): Jacobian matrix of the system.
    err_vector (numpy.ndarray): Error vector representing the desired task-space velocity.
    mode (str): Operation mode, either "moveL" (moving to a target point) or "moveu" (moving with a given velocity).

    Returns:
    numpy.ndarray: Computed joint velocity vector.
    """

    # Define a reference joint posture for null space optimization
    q_rest = np.array([-1.68329992, -1.84052448, 0.53755268,  0.84323017, -0.1275651,  -1.23642077,
                        1.66654381,  2.72159296, -2.45620973,  4.72680824,  0.70164811,  0.13023312])
    
    # q has 12 dimension, while qd has 11(if including v_y)?
    # Exclude the first two joints -- one is planar_y, but idk what another joint is
    q = q[2:]
    q_rest = q_rest[2:]

    # Mask to specify which joints should be optimized in null space motion
    mask = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])

    # Select joint weights based on mode
    if mode == "moveL":
        # Weights for moving to a target position
        joint_weights = np.array([1, 0.1, 1, 1, 1, 1, 1, 0.1, 1, 1])   
    elif mode == "moveu":
        # Weights for moving with a given velocity
        joint_weights = np.array([10, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1])
    else:
        raise ValueError("Invalid mode. Use 'moveL' or 'moveu'.")

    # Construct the weight matrix
    W = np.diag(joint_weights)

    # Compute the adaptive damping factor based on manipulability
    manipulability = compute_manipulability(J[:, 2:])  # Compute manipulability only for the arm joints
    lambda_adaptive = tikhonov_damp / manipulability

    # Compute the damped weighted pseudo-inverse
    JWJ = J @ np.linalg.inv(W) @ J.T + lambda_adaptive * np.eye(J.shape[0])
    J_pseudo = np.linalg.inv(W) @ J.T @ np.linalg.inv(JWJ)

    # Compute the primary task velocity
    qd_task = J_pseudo @ err_vector

    # If in "moveu" mode, return only the primary task velocity
    if mode == "moveu":
        return qd_task

    # Compute the null space projection matrix
    I = np.eye(J.shape[1])
    N = I - J_pseudo @ J  # Null space projection matrix

    # Compute optimization direction for selected joints
    z = - (q - q_rest)  # Negative gradient direction, moving selected joints toward q_rest
    z *= mask  # Apply mask to optimize only specific joints
    qd_null = N @ z  # Project onto null space

    # Combine primary task and null space motion
    qd = qd_task + qd_null

    return qd

def getClikController(args, robot):
    """
    getClikController
    -----------------
    A string argument is used to select one of these.
    It's a bit ugly, bit totally functional and OK solution.
    we want all of theme to accept the same arguments, i.e. the jacobian and the error vector.
    if they have extra stuff, just map it in the beginning with partial
    NOTE: this could be changed to something else if it proves inappropriate later
    TODO: write out other algorithms
    """
    if args.clik_controller == "dampedPseudoinverse":
        return partial(dampedPseudoinverse, args.tikhonov_damp)
    if args.clik_controller == "dPi_adptive_manipulability":
        return partial(dPi_adptive_manipulability, args.tikhonov_damp)
    if args.clik_controller == "dPi_Weighted":
        return partial(dPi_Weighted, args.tikhonov_damp)
    if args.clik_controller == "taskPriorityInverse":
        return partial(taskPriorityInverse, args.tikhonov_damp)
    
    return partial(dampedPseudoinverse, args.tikhonov_damp)


def controlLoopClik(robot: RobotManager, clik_controller, i, past_data):
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    # first check whether we're at the goal
    SEerror = T_w_e.actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector
    if np.linalg.norm(err_vector) < robot.args.goal_error:
        breakFlag = True
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    J = np.delete(J, 1, axis=1)
    # compute the joint velocities based on controller you passed
    # qd = clik_controller(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    if robot.args.clik_controller == "dPi_Weighted":
        qd = dPi_Weighted(1e-3, J, err_vector, "moveL")
    elif robot.args.clik_controller == "dPi_Weighted_nullspace":
        qd = dPi_Weighted_nullspace(1e-3, q, J, err_vector, "moveL")
    else:
        qd = clik_controller(J, err_vector)
    qd = np.insert(qd, 1, 0)
    # print(qd)
    if qd is None:
        print("the controller you chose didn't work, using dampedPseudoinverse instead")
        qd = dampedPseudoinverse(1e-2, J, err_vector)
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    log_item["err_norm"] = np.linalg.norm(err_vector).reshape((1,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    save_past_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    return breakFlag, save_past_item, log_item


def controlLoopClikDualArm(
    robot: RobotManager, clik_controller, goal_transform, i, past_data
):
    """
    controlLoopClikDualArm
    ---------------
    do point to point motion for each arm and its goal.
    that goal is generated from a single goal that you pass,
    and a transformation on the goal you also pass.
    the transformation is done in goal's frame (goal is and has
    to be an SE3 object).
    the transform is designed for the left arm,
    and its inverse is applied for the right arm.
    """
    breakFlag = False
    log_item = {}
    q = robot.getQ()
    T_w_e_left, T_w_e_right = robot.getT_w_e()
    #
    Mgoal_left = goal_transform.act(robot.Mgoal)
    Mgoal_right = goal_transform.inverse().act(robot.Mgoal)
    # print("robot.Mgoal", robot.Mgoal)
    # print("Mgoal_left", Mgoal_left)
    # print("Mgoal_right", Mgoal_right)

    SEerror_left = T_w_e_left.actInv(Mgoal_left)
    SEerror_right = T_w_e_right.actInv(Mgoal_right)

    err_vector_left = pin.log6(SEerror_left).vector
    err_vector_right = pin.log6(SEerror_right).vector

    if (np.linalg.norm(err_vector_left) < robot.args.goal_error) and (
        np.linalg.norm(err_vector_right) < robot.args.goal_error
    ):
        breakFlag = True
    J_left = pin.computeFrameJacobian(robot.model, robot.data, q, robot.l_ee_frame_id)
    J_right = pin.computeFrameJacobian(robot.model, robot.data, q, robot.r_ee_frame_id)
    # compute the joint velocities based on controller you passed
    # this works exactly the way you think it does:
    # the velocities for the other arm are 0
    # what happens to the base hasn't been investigated but it seems ok
    qd_left = clik_controller(J_left, err_vector_left)
    qd_right = clik_controller(J_right, err_vector_right)
    qd = qd_left + qd_right
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    return breakFlag, {}, log_item

def moveL(args, robot: RobotManager, goal_point):
    """
    moveL
    -----
    does moveL.
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    # assert type(goal_point) == pin.pinocchio_pywrap.SE3
    robot.Mgoal = copy.deepcopy(goal_point)
    clik_controller = getClikController(args, robot)
    controlLoop = partial(controlLoopClik, robot, clik_controller)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "dqs_cmd": np.zeros(robot.model.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {
        "dqs_cmd": np.zeros(robot.model.nv),
    }
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()

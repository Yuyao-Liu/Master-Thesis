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
from scipy.spatial.transform import Rotation as R
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

def compute_manipulability(J):
    # print(np.linalg.det(J @ J.T))
    return np.sqrt(np.linalg.det(J @ J.T) + np.eye(J.shape[0], J.shape[0]) * 1e-6)

def add_bias_and_noise(vec):
    noise_std=0.01
    bias = np.random.uniform(low=-0.5, high=0.5, size=vec.shape)
    vec = np.asarray(vec)
    bias = np.asarray(bias) if isinstance(bias, (list, np.ndarray)) else bias
    noise = np.random.normal(loc=0.0, scale=noise_std, size=vec.shape)
    return vec + bias + noise

# Transform a velocity vector from the world frame to the end-effector (EE) frame
def transform_velocity_to_e(robot: RobotManager):
    """
    Transform a velocity vector from the world frame to the end-effector (EE) frame.

    Parameters:
    - T_w_e: SE3 transformation matrix (Pinocchio SE3 object), representing the pose of the end-effector in the world frame.
    - u_ref_w: 6D velocity vector [v_x, v_y, v_z, ω_x, ω_y, ω_z] in the world frame.

    Returns:
    - u_ref_e: 6D velocity vector in the EE frame.
    """
    # Extract the rotation matrix R_w_e (3x3)
    T_w_e = robot.getT_w_e()
    R_w_e = T_w_e.rotation  

    # Extract linear and angular velocity components in the world frame
    u_ref_w = robot.u_ref_w
    v_w = u_ref_w[:3]  # Linear velocity in the world frame
    omega_w = u_ref_w[3:]  # Angular velocity in the world frame

    # Transform velocities to the EE frame
    v_e = R_w_e.T @ v_w  # Linear velocity in the EE frame
    omega_e = R_w_e.T @ omega_w  # Angular velocity in the EE frame

    # Combine into a 6D velocity vector
    u_ref_e = np.hstack((v_e, omega_e))
    
    return u_ref_e

def compute_ee2basedistance(robot: RobotManager):
    q = robot.getQ()
    x_base = q[0]
    y_base = q[1]
    T_w_e = robot.getT_w_e()
    x_ee = T_w_e.translation[0]
    y_ee = T_w_e.translation[1]
    d = np.sqrt((x_base-x_ee)**2+(y_base-y_ee)**2)
    return d

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

def dPi_Weighted_nullspace(tikhonov_damp, q, J, err_vector, mode, robot):
    """
    Computes the weighted damped pseudo-inverse with null space motion optimization.
    Ensures that the end-effector (EE) maintains a minimum base distance (>= 0.5) in "moveu" mode.

    Parameters:
    ----------
    tikhonov_damp : float
        Regularization damping factor for numerical stability.
    q : numpy.ndarray
        Current joint positions.
    J : numpy.ndarray
        Jacobian matrix of the robot.
    err_vector : numpy.ndarray
        Error vector representing the desired task-space velocity.
    mode : str
        Operation mode, either:
        - "moveL": Moves towards a target position.
        - "moveu": Moves with a given velocity while maintaining a minimum base distance.
    robot : RobotManager
        Robot instance, used to compute the end-effector to base distance.

    Returns:
    -------
    numpy.ndarray
        Computed joint velocity vector.
    """

    # Rest posture of the robot
    q_rest = np.array([-1.40905799, -2.29935204,  0.96482552,  0.26289107, -0.12981707, -1.2151303,
                        1.62831697, -0.41679404,  1.70733658,  1.57549959, -0.2871302,  -0.45769692])

    # Remove y and z degrees of freedom
    q = np.delete(q, [1, 2])
    q_rest = np.delete(q_rest, [1, 2])

    # Define a mask to optimize only specific joints
    # mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    mask = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    # Assign different joint weights based on the mode
    if mode == "moveL":
        joint_weights = np.array([1, 0.1, 1, 1, 1, 1, 1, 0.1, 1, 1])
    elif mode == "moveu":
        joint_weights = np.array([10, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1])
    else:
        raise ValueError("Invalid mode. Use 'moveL' or 'moveu'.")

    W = np.diag(joint_weights)

    # Compute adaptive damping factor
    manipulability = compute_manipulability(J[:, 2:])
    lambda_adaptive = tikhonov_damp / manipulability

    # Compute the weighted damped pseudo-inverse
    JWJ = J @ np.linalg.inv(W) @ J.T + lambda_adaptive * np.eye(J.shape[0])
    J_pseudo = np.linalg.inv(W) @ J.T @ np.linalg.inv(JWJ)

    # Compute primary task velocity
    qd_task = J_pseudo @ err_vector

    d_min = 0.7  # Minimum allowed EE-base distance

    if mode == "moveu":
        d = compute_ee2basedistance(robot)

        # If the EE-base distance is sufficient, return the computed velocity directly
        if d >= d_min:
            return qd_task

        # # Compute the retreating direction for base movement
        # base_pos = robot.getT_w_e().translation[:2]  # X, Y position of base
        # ee_pos = q[:2]  # X, Y position of end-effector
        # direction_vector = ee_pos - base_pos
        # direction_vector /= np.linalg.norm(direction_vector)  # Normalize
        # move_direction = np.sign(direction_vector[1])  # Determine whether to move forward or backward

        # Compute null space projection
        I = np.eye(J.shape[1])
        N = I - J_pseudo @ J
        z = np.zeros_like(qd_task)
        z[0] = -0.8 * (d - d_min) # Control the base to retreat
        qd_null = N @ z

        # Combine primary task velocity and null space velocity
        return qd_task + qd_null

    # Compute the null space projection matrix
    I = np.eye(J.shape[1])
    N = I - J_pseudo @ J

    # Compute the null space direction
    z = -(q - q_rest)
    z *= mask
    qd_null = N @ z
    # Combine primary task velocity and null space velocity
    return qd_task + qd_null

def parking_base(q, target_pose):
    """
    Compute the linear velocity (v) and angular velocity (omega) for a differential drive robot.
    
    Parameters:
    - q: Robot state, where q[0] and q[1] represent the x and y coordinates, and q[2], q[3] are quaternion values.
    - target_pose: (x, y, theta) Target position in meters and radians.

    Returns:
    - qd: An array containing the computed velocities.
    """

    # Control gains (adjustable)
    k1 = 1.0  # Linear velocity gain
    k2 = 2.0  # Angular velocity gain
    k3 = 1.0  # Orientation error gain

    # Extract robot's current pose
    x_r, y_r, theta_r = (q[0], q[1], np.arctan2(q[3], q[2]))  
    x_t, y_t, theta_t = target_pose  

    # Compute the relative position between robot and target
    dx = x_r - x_t
    dy = y_r - y_t
    rho = np.hypot(dx, dy)  # Euclidean distance to the target

    # Calculate the target's bearing angle in the global frame
    theta_target = np.arctan2(dy, dx)

    # Compute the bearing error in the robot's local frame
    gamma = theta_target - theta_r + np.pi  # Adjust for orientation

    # Normalize the angle to (-pi, pi] range
    gamma = (gamma + np.pi) % (2 * np.pi) - np.pi

    # Compute the final orientation error
    delta = gamma + theta_r - theta_t
    delta = (delta + np.pi) % (2 * np.pi) - np.pi  # Normalize again

    # Compute linear velocity
    v = k1 * rho * np.cos(gamma)

    # Compute angular velocity, avoiding division by zero
    if gamma == 0:
        omega = 0  
    else:
        omega = k2 * gamma + k1 * (np.sin(gamma) * np.cos(gamma) / gamma) * (gamma + k3 * delta)

    # Construct the velocity output array
    qd = np.array([v, 0, omega, 0, 0, 0, 0, 0, 0, 0, 0])
    return qd


def keep_distance_nullspace(tikhonov_damp, q, J, err_vector, robot):
    q = add_bias_and_noise(q)
    (x_base, y_base, theta_base) = (q[0], q[1], np.arctan2(q[3], q[2]))
    T_w_e = robot.getT_w_e()
    (x_ee, y_ee) = (T_w_e.translation[0], T_w_e.translation[1])
    
    J_w = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
    # joint_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # W = np.diag(joint_weights)

    # # Compute adaptive damping factor
    # manipulability = compute_manipulability(J[:, 2:])
    # lambda_adaptive = tikhonov_damp / manipulability
    # # Compute the weighted damped pseudo-inverse
    # JWJ = J @ np.linalg.inv(W) @ J.T + lambda_adaptive * np.eye(J.shape[0])
    # J_pseudo = np.linalg.inv(W) @ J.T @ np.linalg.inv(JWJ)
    
    J_pseudo = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * tikhonov_damp)
    # Compute primary task velocity
    qd_task = J_pseudo @ err_vector

    ### compute q_null ###
    d_target = 0.6  # Minimum allowed EE-base distance
    dx = x_ee - x_base
    dy = y_ee - y_base
    d_current = np.hypot(dx, dy)
    print(d_current)
    I = np.eye(J.shape[1])
    N = I - J_pseudo @ J
    
    Jx = J[0, :]
    Jy = J[1, :]
    Jbx = np.zeros_like(Jx)
    Jbx[0] = 1
    Jd = (dx * (Jx - Jbx) + dy * Jy)/d_current
    # z1 = -5 * Jd.T * np.sign(d_current - d_target)
    z1 = -20 * Jd.T * (d_current - d_target)
    # print(Jx,Jy)
    # print(z1)
    
    z2 = np.zeros_like(z1)
    
    J_w = np.delete(J_w,1,axis=1)
    xd = J_w @ qd_task
    
    # qd = robot.getQd()
    # xd = J_w @ qd
    # print(xd)
    dir_vee = np.array([xd[0], xd[1]])
    dir_base = np.array([q[2], q[3]])
    dir_eb = np.array([dx, dy])
    
    ## b to a counterclockwise
    def angle_between_vectors(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        theta_a = np.arctan2(a[1], a[0])
        theta_b = np.arctan2(b[1], b[0])
        angle = theta_a - theta_b
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        # dot_product = np.dot(a, b)
        # dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # angle = np.arccos(dot_product)
        
        # cross_product = np.cross(a, b)
        
        # if cross_product < 0:
        #     angle = -angle
        if angle > np.pi/2:
            angle = angle - np.pi
        if angle < -np.pi/2:
            angle = angle + np.pi
        return angle
    dir_e_z = np.array([T_w_e.rotation[0, 2], T_w_e.rotation[0, 1]])
    theta = angle_between_vectors(dir_vee, dir_base)
    # theta = angle_between_vectors(dir_e_z, dir_base)
    z2[1] = 0.5 * (theta)
    # z2[2] = -z2[1]
    # print(z2[1])
    # print(q[4])
    # if np.abs(d_current - d_target) < 0.05:
    #     qd_null = N @ (z1 + z2)
    #     # qd_null = N @ z2
    # else:
    #     qd_null = N @ z1
    qd_null = N @ (z1 + z2)
    # Combine primary task velocity and null space velocity
    return qd_task + qd_null


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


# controlLoopClik_u_ref for move_u_ref
def controlLoopClik_u_ref(robot: RobotManager, Adaptive_controller, clik_controller, i, past_data):
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.getQ()
    # x, y, z, omega, q_1, q_2, q_3, q_4, q_5, q_6, g_1, g_2
    # print(q)
    # TODO set a proper omega
    v_ref = Adaptive_controller.get_v_ref()
    robot.u_ref_w = np.hstack((v_ref, np.zeros(3)))
    # print(robot.u_ref_w)
    
    # Convert the twist u_ref_w (6D) in world frame to ee frame
    # u_ref_e = transform_velocity_to_e(robot)
    # print(u_ref_e)
    # err_vector = u_ref_e
    v = -np.pi/40
    R = 1
    mode = 1
    if mode == 1:
        # open a revolving door
        err_vector = np.array([0, 0, v, -v/R, 0, 0])
    elif mode == 2:
        # open a revolving drawer
        err_vector = np.array([0, 0, v, 0, -v/R, 0])
    elif mode == 3:
        # open a sliding door
        err_vector = np.array([0, -v, 0, 0, 0, 0])
    elif mode == 4:
        # open a sliding drawer
        err_vector = np.array([0, 0, v, 0, 0, 0])

    # err_vector = robot.u_ref_w
    
    # J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)
    # print(J)
    # delete the second columm of Jacobian matrix, cuz y_dot is always 0
    # J[:, 1] = 1e-6
    J = np.delete(J, 1, axis=1)
    
    # compute the joint velocities based on controller you passed
    if robot.args.clik_controller == "dPi_Weighted":
        qd = dPi_Weighted(1e-3, J, err_vector, "moveu")
    elif robot.args.clik_controller == "dPi_Weighted_nullspace":
        qd = dPi_Weighted_nullspace(1e-3, q, J, err_vector, "moveu", robot)
    elif robot.args.clik_controller == "keep_distance_nullspace":
        qd = keep_distance_nullspace(1e-3, q, J, err_vector, robot)
    else:
        qd = clik_controller(J, err_vector)
            
    qd = np.insert(qd, 1, 0)
    # v_x, v_y, omega, q_1, q_2, q_3, q_4, q_5, q_6,
    # qd = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    robot.sendQd(qd)
    err_vector = compute_ee2basedistance(robot)
    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    log_item["err_norm"] = np.linalg.norm(err_vector).reshape((1,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    save_past_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    return breakFlag, save_past_item, log_item

def controlLoopClik_park(robot: RobotManager, clik_controller, target_pose, i, past_data):
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.getQ()
    qd = clik_controller(q, target_pose)
    if np.linalg.norm(np.array(target_pose)-[q[0], q[1], np.arctan2(q[3], q[2])]) < robot.args.goal_error:
        breakFlag = True
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    log_item["err_norm"] = np.linalg.norm(np.zeros(1)).reshape((1,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    save_past_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    return breakFlag, save_past_item, log_item

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
    # print(T_w_e)
    # first check whether we're at the goal
    SEerror = T_w_e.actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector
    if np.linalg.norm(err_vector) < robot.args.goal_error:
        breakFlag = True
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    J = J[:, 3:]
    # compute the joint velocities based on controller you passed
    # qd = clik_controller(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    qd = clik_controller(J, err_vector)
    qd = np.concatenate((np.zeros(3), qd))
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

def move_u_ref(args, robot: RobotManager, Adaptive_controller):
    """
    move_u_ref
    -----
    come from moveL
    send a reference twist u_ref_w instead.
    """
    clik_controller = getClikController(args, robot)
    controlLoop = partial(controlLoopClik_u_ref, robot, Adaptive_controller, clik_controller)
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
    
def park_base(args, robot: RobotManager, target_pose):
    controlLoop = partial(controlLoopClik_park, robot, parking_base, target_pose)
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
    

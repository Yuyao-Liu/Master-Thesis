from smc.control.control_loop_manager import ControlLoopManager
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.robots.interfaces.dual_arm_interface import DualArmInterface
from smc.control.controller_templates.point_to_point import (
    EEP2PCtrlLoopTemplate,
    DualEEP2PCtrlLoopTemplate,
)
from smc.robots.interfaces.force_torque_sensor_interface import (
    ForceTorqueOnSingleArmWrist,
)
from smc.control.cartesian_space.ik_solvers import *
from functools import partial
import pinocchio as pin
import numpy as np
from argparse import Namespace
from collections import deque
from typing import Callable
from scipy.spatial.transform import Rotation as R
from smc.control.cartesian_space.ik_solvers import QPManipMax


def controlLoopClik(
    #                       J           err_vec     v_cmd
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    T_w_goal: pin.SE3,
    args: Namespace,
    robot: SingleArmInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    T_w_e = robot.T_w_e
    SEerror = T_w_e.actInv(T_w_goal)
    err_vector = pin.log6(SEerror).vector
    J = robot.getJacobian()
    # compute the joint velocities based on controller you passed
    # qd = ik_solver(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    if args.ik_solver == "QPManipMax":
        v_cmd = QPManipMax(
            J,
            err_vector,
            robot.computeManipulabilityIndexQDerivative(),
            lb=-1 * robot.max_v,
            ub=robot.max_v,
        )
    else:
        v_cmd = ik_solver(J, err_vector)
    if v_cmd is None:
        print(
            t,
            "the controller you chose produced None as output, using dampedPseudoinverse instead",
        )
        v_cmd = dampedPseudoinverse(1e-2, J, err_vector)
    else:
        if args.debug_prints:
            print(t, "ik solver success")

    return v_cmd, {}, {}

def controlLoopClik_only_arm(
    #                       J           err_vec     v_cmd
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    T_w_goal: pin.SE3,
    args: Namespace,
    robot: SingleArmInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    T_w_e = robot.T_w_e
    SEerror = T_w_e.actInv(T_w_goal)
    err_vector = pin.log6(SEerror).vector
    J = robot.getJacobian()
    J = J[:, 3:]
    # compute the joint velocities based on controller you passed
    # qd = ik_solver(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    if args.ik_solver == "QPManipMax":
        v_cmd = QPManipMax(
            J,
            err_vector,
            robot.computeManipulabilityIndexQDerivative(),
            lb=-1 * robot.max_v,
            ub=robot.max_v,
        )
    else:
        v_cmd = ik_solver(J, err_vector)
    if v_cmd is None:
        print(
            t,
            "the controller you chose produced None as output, using dampedPseudoinverse instead",
        )
        v_cmd = dampedPseudoinverse(1e-2, J, err_vector)
    else:
        if args.debug_prints:
            print(t, "ik solver success")
    v_cmd = np.concatenate((np.zeros(3), v_cmd))
    return v_cmd, {}, {}

def controlLoopClik_park(robot, clik_controller, target_pose, i, past_data):
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q
    v_cmd = clik_controller(q, target_pose)
    if np.linalg.norm(np.array(target_pose)-[q[0], q[1], np.arctan2(q[3], q[2])]) < robot.args.goal_error:
        breakFlag = True
    robot.sendVelocityCommand(v_cmd)

    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {}
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    return breakFlag, save_past_item, log_item

def park_base(
    args: Namespace, robot: SingleArmInterface, target_pose: pin.SE3, run=True
) -> None | ControlLoopManager:
    
    # assert type(T_w_goal) == pin.SE3
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
    if run:
        loop_manager.run()
    else:
        return loop_manager

def moveL_only_arm(
    args: Namespace, robot: SingleArmInterface, T_w_goal: pin.SE3, run=True
) -> None | ControlLoopManager:
    """
    moveL
    -----
    does moveL.
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    assert type(T_w_goal) == pin.SE3
    ik_solver = getIKSolver(args, robot)
    controlLoop = partial(
        EEP2PCtrlLoopTemplate, ik_solver, T_w_goal, controlLoopClik_only_arm, args, robot
    )
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

def rotate_point_and_orientation(handle_pose, axis_point, axis_direction, angle_deg):
    (pos, orientation) = (handle_pose.translation, handle_pose.rotation)
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    rot = R.from_rotvec(np.deg2rad(angle_deg) * axis_direction)

    pos_relative = pos - axis_point

    pos_rotated = rot.apply(pos_relative)

    new_translation = pos_rotated + axis_point

    new_orientation = rot.as_matrix() @ orientation
    new_pose = pin.SE3(new_orientation, new_translation)
    
    return new_pose

def compute_rotated_angle(handle_pose, T_w_e, axis_point, axis_direction):
    pos_init = handle_pose.translation
    pos_current = T_w_e.translation
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    v1 = pos_init - axis_point
    v2 = pos_current - axis_point

    v1_proj = v1 - np.dot(v1, axis_direction) * axis_direction
    v2_proj = v2 - np.dot(v2, axis_direction) * axis_direction

    v1_proj /= np.linalg.norm(v1_proj)
    v2_proj /= np.linalg.norm(v2_proj)

    cos_theta = np.clip(np.dot(v1_proj, v2_proj), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.rad2deg(angle_rad)

    cross = np.cross(v1_proj, v2_proj)
    if np.dot(cross, axis_direction) < 0:
        angle_deg = -angle_deg

    return angle_deg

def move_u_ref(args: Namespace, robot: SingleArmInterface, Adaptive_controller, run=True):
    """
    move_u_ref
    -----
    come from moveL
    send a reference twist u_ref_w instead.
    """
    new_pose = rotate_point_and_orientation(robot.handle_pose, np.array([-2.0,-1.7,0.5]), np.array([0,0,1]), robot.angle_desired)
    if args.visualizer:
        robot.visualizer_manager.sendCommand({"Mgoal": new_pose})
    controlLoop = partial(controlLoopClik_u_ref, robot, Adaptive_controller, new_pose)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

def controlLoopClik_u_ref(robot: SingleArmInterface, Adaptive_controller, new_pose, i, past_data): 
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q
    # x, y, z, omega, q_1, q_2, q_3, q_4, q_5, q_6, g_1, g_2
    # print(q)
    # TODO set a proper omega
    # v_ref = Adaptive_controller.get_v_ref()
    # robot.u_ref_w = np.hstack((v_ref, np.zeros(3)))
    # print(robot.u_ref_w)
    
    # Convert the twist u_ref_w (6D) in world frame to ee frame
    # u_ref_e = transform_velocity_to_e(robot)
    # print(u_ref_e)
    # err_vector = u_ref_e
    
    angle_moved = compute_rotated_angle(robot.handle_pose, robot.T_w_e, axis_point = np.array([-2.0,-1.7,0.5]), axis_direction = np.array([0,0,1]))
    K = (robot.angle_desired - angle_moved)
    # if K < 1e-5:
    #     Adaptive_controller.save_history_to_mat("log.mat")
    #     breakFlag = True
    
    v_max = np.pi/40
    # v = np.clip(K * v_max, -v_max, v_max)
    v = v_max
    robot.v_ee = v
    R = 0.8
    mode = 1
    if mode == 1:
        # open a revolving door
        err_vector = np.array([0, 0, -v, v/R, 0, 0])
    elif mode == 2:
        # open a revolving drawer
        err_vector = np.array([0, 0, -v, 0, v/R, 0])
    elif mode == 3:
        # open a sliding door
        err_vector = np.array([0, v, 0, 0, 0, 0])
    elif mode == 4:
        # open a sliding drawer
        err_vector = np.array([0, 0, -v, 0, 0, 0])
    elif mode == 5:
        # open a sliding drawer
        err_vector = np.array([0, 0, 0, 1, 0, 0])
    x_h_oe = Adaptive_controller.get_x_h()
    # Adaptive_controller.save_history_to_mat("log.mat")
    # print(x_h_oe)
    # err_vector = robot.u_ref_w
    
    # J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)
    # print(J)
    # delete the second columm of Jacobian matrix, cuz y_dot is always 0
    # J[:, 1] = 1e-6
    
    
    # compute the joint velocities based on controller you passed
    v_cmd = keep_distance_nullspace(1e-3, q, J, err_vector, robot)
    robot.sendVelocityCommand(v_cmd)
    # v_x, v_y, omega, q_1, q_2, q_3, q_4, q_5, q_6,
    # qd = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    return breakFlag, save_past_item, log_item


# TODO: implement
def moveLFollowingLine(
    args: Namespace, robot, goal_point: pin.SE3
) -> tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    moveLFollowingLine
    ------------------
    make a path from current to goal position, i.e.
    just a straight line between them.
    the question is what to do with orientations.
    i suppose it makes sense to have one function that enforces/assumes
    that the start and end positions have the same orientation.
    then another version goes in a line and linearly updates the orientation
    as it goes
    """
    ...


def moveUntilContactControlLoop(
    args: Namespace,
    robot: ForceTorqueOnSingleArmWrist,
    speed: np.ndarray,
    #                       J           err_vec     v_cmd
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    i: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    moveUntilContactControlLoop
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    breakFlag = False
    # know where you are, i.e. do forward kinematics
    log_item = {}
    q = robot.q
    # break if wrench is nonzero basically
    # wrench = robot.getWrench()
    # you're already giving the speed in the EE i.e. body frame
    # so it only makes sense to have the wrench in the same frame
    # wrench = robot._getWrenchInEE()
    wrench = robot.wrench
    # and furthermore it's a reasonable assumption that you'll hit the thing
    # in the direction you're going in.
    # thus we only care about wrenches in those direction coordinates
    mask = speed != 0.0
    # NOTE: contact getting force is a magic number
    # it is a 100% empirical, with the goal being that it's just above noise.
    # so far it's worked fine, and it's pretty soft too.
    if np.linalg.norm(wrench[mask]) > args.contact_detecting_force:
        print("hit with", np.linalg.norm(wrench[mask]))
        breakFlag = True
        robot.sendVelocityCommand(np.zeros(robot.nv))
    if (not args.real) and (i > 500):
        print("let's say you hit something lule")
        breakFlag = True
    # pin.computeJointJacobian is much different than the C++ version lel
    J = robot.getJacobian()
    # compute the joint velocities.
    qd = ik_solver(J, speed)
    robot.sendVelocityCommand(qd)
    log_item["qs"] = q.reshape((robot.nq,))
    log_item["wrench"] = wrench.reshape((6,))
    return breakFlag, {}, log_item


def moveUntilContact(
    args: Namespace, robot: ForceTorqueOnSingleArmWrist, speed: np.ndarray
) -> None:
    """
    moveUntilContact
    -----
    does clik until it feels something with the f/t sensor
    """
    assert type(speed) == np.ndarray
    ik_solver = getIKSolver(args, robot)
    controlLoop = partial(moveUntilContactControlLoop, args, robot, speed, ik_solver)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {"wrench": np.zeros(6)}
    log_item["qs"] = np.zeros((robot.nq,))
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()
    print("Collision detected!!")


def controlLoopClikDualArm(
    #                       J           err_vec     v_cmd
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    T_w_absgoal: pin.SE3,
    T_absgoal_l: pin.SE3,
    T_absgoal_r: pin.SE3,
    args: Namespace,
    robot: DualArmInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopClikDualArm
    ---------------
    do point to point motion for each arm and its goal.
    that goal is generated from a single goal that you pass,
    and an SE3  transformation on the goal for each arm
    """

    T_w_lgoal = T_absgoal_l.act(T_w_absgoal)
    T_w_rgoal = T_absgoal_r.act(T_w_absgoal)

    SEerror_left = robot.T_w_l.actInv(T_w_lgoal)
    SEerror_right = robot.T_w_r.actInv(T_w_rgoal)

    err_vector_left = pin.log6(SEerror_left).vector
    err_vector_right = pin.log6(SEerror_right).vector

    err_vector = np.concatenate((err_vector_left, err_vector_right))
    J = robot.getJacobian()

    if args.ik_solver == "QPManipMax":
        v_cmd = QPManipMax(
            J,
            err_vector,
            robot.computeManipulabilityIndexQDerivative(),
            lb=-1 * robot.max_v,
            ub=robot.max_v,
        )
    else:
        v_cmd = ik_solver(J, err_vector)
    if v_cmd is None:
        print(
            t,
            "the controller you chose produced None as output, using dampedPseudoinverse instead",
        )
        v_cmd = dampedPseudoinverse(1e-2, J, err_vector)
    else:
        if args.debug_prints:
            print(t, "ik solver success")
    return v_cmd, {}, {}


def moveLDualArm(
    args: Namespace,
    robot: DualArmInterface,
    T_w_goal: pin.SE3,
    T_abs_l: pin.SE3,
    T_abs_r: pin.SE3,
    run=True,
) -> None | ControlLoopManager:
    """
    moveLDualArm
    -----------
    """
    ik_solver = getIKSolver(args, robot)
    controlLoop = partial(
        DualEEP2PCtrlLoopTemplate,
        ik_solver,
        T_w_goal,
        T_abs_l,
        T_abs_r,
        controlLoopClikDualArm,
        args,
        robot,
    )
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "l_err_norm": np.zeros(1),
        "r_err_norm": np.zeros(1),
    }
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

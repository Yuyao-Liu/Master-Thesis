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
import time
from scipy.io import savemat

q_park = []
q_pull = []
T_w_e_pull = []
rotation_pull = []
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
    global q_park
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q
    q_park.append(q.copy())
    # print(q)
    v_cmd = clik_controller(q, target_pose)
    # v_cmd = np.array([0,0,0.1,0,0,0,0,0,0])
    # v_cmd = np.zeros(robot.nv)
    # v_cmd[2]=1
    current_error = np.linalg.norm(target_pose-np.array([q[0], q[1], np.arctan2(q[3], q[2])]))
    if current_error < robot.args.goal_error:
        breakFlag = True
        savemat("q_park.mat", {"q_park": np.array(q_park)})
        print("q_park saved")
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
    args: Namespace, robot: SingleArmInterface, target_pose, run=True
) -> None | ControlLoopManager:
    # time.sleep(5)
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
    # time.sleep(2)
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
    # time.sleep(2)
    Adaptive_controller.update_time()
    """
    move_u_ref
    -----
    come from moveL
    send a reference twist u_ref_w instead.
    """
    new_pose = rotate_point_and_orientation(robot.handle_pose, np.array([-2.3, -0.65-0.8, 1]), np.array([0,0,1]), robot.angle_desired)
    # if args.visualizer:
    #     robot.visualizer_manager.sendCommand({"Mgoal": new_pose})
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
    global q_pull
    global T_w_e_pull
    global rotation_pull
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q
    T_w_e = robot.T_w_e.translation
    rotation = robot.T_w_e.rotation
    rotation_pull.append(rotation.copy())
    T_w_e_pull.append(T_w_e.copy())
    q_pull.append(q.copy())
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
    
    angle_moved = compute_rotated_angle(robot.handle_pose, robot.T_w_e, axis_point = np.array([-2.3, -0.65-0.8, 1]), axis_direction = np.array([0,0,1]))
    # print(angle_moved)
    # K = abs(angle_moved)
    # if len(q_pull) == 0.8e4:
    #     breakFlag = True
    #     savemat("q_pull.mat", {"q_pull": np.array(q_pull)})
    #     savemat("T_w_e_pull.mat", {"T_w_e_pull": np.array(T_w_e_pull)})
    #     savemat("rotation_pull.mat", {"rotation_pull": np.array(rotation_pull)})
    #     print("q_pull, rotation_pull and T_w_e_pull saved")
    
    v_max = np.pi/40
    # v = np.clip(K * v_max, -v_max, v_max)
    v = v_max
    robot.v_ee = v
    mode = robot.task
    if mode == 1:
        R = 0.8
        # open a revolving door
        err_vector = np.array([0, 0, -v, v/R, 0, 0])
    elif mode == 2:
        R = 0.5
        # open a revolving drawer
        err_vector = np.array([0, 0, -v, 0, -v/R, 0])
    elif mode == 3:
        # open a sliding door
        err_vector = np.array([0, v, 0, 0, 0, 0])
    elif mode == 4:
        # open a sliding drawer
        err_vector = np.array([0, 0, -v, 0, 0, 0])
    x_h_oe = Adaptive_controller.get_x_h()
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
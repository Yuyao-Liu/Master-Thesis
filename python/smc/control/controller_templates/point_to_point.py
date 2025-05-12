from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.robots.interfaces.dual_arm_interface import DualArmInterface

from pinocchio import SE3, log6
from argparse import Namespace
from typing import Any, Callable
import numpy as np
from collections import deque

from smc.robots.interfaces.whole_body_dual_arm_interface import (
    DualArmWholeBodyInterface,
)
from smc.robots.interfaces.whole_body_single_arm_interface import (
    SingleArmWholeBodyInterface,
)

global control_loop_return
control_loop_return = tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]


def BaseP2PCtrlLoopTemplate(
    SOLVER: Any,
    p_basegoal: np.ndarray,
    control_loop: Callable[
        [
            Any,
            np.ndarray,
            Namespace,
            MobileBaseInterface,
            int,
            dict[str, deque[np.ndarray]],
        ],
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    ],
    args: Namespace,
    robot: MobileBaseInterface,
    t: int,  # will be float eventually
    past_data: dict[str, deque[np.ndarray]],
) -> control_loop_return:
    """
    EEAndBaseP2PCtrlLoopTemplate
    ---------------
    generic control loop for point to point motion for end-effectors of a dual arm robot
    (handling error to final point etc).
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}

    v_cmd, past_item_inner, log_item_inner = control_loop(
        SOLVER, p_basegoal, args, robot, t, past_data
    )
    robot.sendVelocityCommand(v_cmd)
    log_item.update(log_item_inner)
    save_past_item.update(past_item_inner)

    p_base = np.array(list(robot.q[:2]) + [0.0])
    base_err_vector_norm = np.linalg.norm(p_basegoal - p_base)

    if base_err_vector_norm < robot.args.goal_error:
        breakFlag = True

    log_item["qs"] = robot.q
    log_item["dqs"] = robot.v
    log_item["dqs_cmd"] = v_cmd.reshape((robot.model.nv,))
    log_item["base_err_norm"] = base_err_vector_norm.reshape((1,))
    return breakFlag, save_past_item, log_item


def EEP2PCtrlLoopTemplate(
    SOLVER: Any,
    T_w_goal: SE3,
    control_loop: Callable[
        [Any, SE3, Namespace, SingleArmInterface, int, dict[str, deque[np.ndarray]]],
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    ],
    args: Namespace,
    robot: SingleArmInterface,
    t: int,  # will be float eventually
    past_data: dict[str, deque[np.ndarray]],
) -> control_loop_return:
    """
    EEP2PCtrlLoopTemplate
    ---------------
    generic control loop for point to point motion with for the end-effector
    (handling error to final point etc).
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}

    v_cmd, past_item_inner, log_item_inner = control_loop(
        SOLVER, T_w_goal, args, robot, t, past_data
    )
    robot.sendVelocityCommand(v_cmd)
    log_item.update(log_item_inner)
    save_past_item.update(past_item_inner)

    T_w_e = robot.T_w_e
    SEerror = T_w_e.actInv(T_w_goal)
    err_vector = log6(SEerror).vector
    err_vector_norm = np.linalg.norm(err_vector)
    if err_vector_norm < robot.args.goal_error:
        breakFlag = True
    log_item["qs"] = robot.q
    log_item["dqs"] = robot.v
    log_item["dqs_cmd"] = v_cmd
    log_item["err_norm"] = err_vector_norm.reshape((1,))
    return breakFlag, save_past_item, log_item


def DualEEP2PCtrlLoopTemplate(
    SOLVER: Any,
    T_w_absgoal: SE3,
    T_abs_l: SE3,
    T_abs_r: SE3,
    control_loop: Callable[
        [
            Any,
            SE3,
            SE3,
            SE3,
            Namespace,
            DualArmInterface,
            int,
            dict[str, deque[np.ndarray]],
        ],
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    ],
    args: Namespace,
    robot: DualArmInterface,
    t: int,  # will be float eventually
    past_data: dict[str, deque[np.ndarray]],
) -> control_loop_return:
    """
    DualEEP2PCtrlLoopTemplate
    ---------------
    generic control loop for point to point motion for end-effectors of a dual arm robot
    (handling error to final point etc).
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}

    v_cmd, past_item_inner, log_item_inner = control_loop(
        SOLVER, T_w_absgoal, T_abs_l, T_abs_r, args, robot, t, past_data
    )
    robot.sendVelocityCommand(v_cmd)
    log_item.update(log_item_inner)
    save_past_item.update(past_item_inner)

    T_w_lgoal = T_abs_l.act(T_w_absgoal)
    T_w_rgoal = T_abs_r.act(T_w_absgoal)

    SEerror_left = robot.T_w_l.actInv(T_w_lgoal)
    SEerror_right = robot.T_w_r.actInv(T_w_rgoal)

    err_vector_left = log6(SEerror_left).vector
    err_vector_right = log6(SEerror_right).vector
    err_vector_left_norm = np.linalg.norm(err_vector_left)
    err_vector_right_norm = np.linalg.norm(err_vector_right)

    if (err_vector_left_norm < robot.args.goal_error) and (
        err_vector_right_norm < robot.args.goal_error
    ):
        breakFlag = True

    log_item["qs"] = robot.q
    log_item["dqs"] = robot.v
    log_item["dqs_cmd"] = v_cmd.reshape((robot.nv,))
    log_item["l_err_norm"] = err_vector_left_norm.reshape((1,))
    log_item["r_err_norm"] = err_vector_right_norm.reshape((1,))
    return breakFlag, save_past_item, log_item


def EEAndBaseP2PCtrlLoopTemplate(
    SOLVER: Any,
    T_w_goal: SE3,
    p_basegoal: np.ndarray,
    control_loop: Callable[
        [
            Any,
            SE3,
            np.ndarray,
            Namespace,
            SingleArmWholeBodyInterface,
            int,
            dict[str, deque[np.ndarray]],
        ],
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    ],
    args: Namespace,
    robot: SingleArmWholeBodyInterface,
    t: int,  # will be float eventually
    past_data: dict[str, deque[np.ndarray]],
) -> control_loop_return:
    """
    EEAndBaseP2PCtrlLoopTemplate
    ---------------
    generic control loop for point to point motion for end-effectors of a dual arm robot
    (handling error to final point etc).
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}

    v_cmd, past_item_inner, log_item_inner = control_loop(
        SOLVER, T_w_goal, p_basegoal, args, robot, t, past_data
    )
    robot.sendVelocityCommand(v_cmd)
    log_item.update(log_item_inner)
    save_past_item.update(past_item_inner)

    T_w_e = robot.T_w_e
    SEerror = T_w_e.actInv(T_w_goal)
    ee_err_vector = log6(SEerror).vector
    ee_err_vector_norm = np.linalg.norm(ee_err_vector)

    p_base = robot.T_w_b.translation
    base_err_vector_norm = np.linalg.norm(p_basegoal - p_base)

    if (ee_err_vector_norm < robot.args.goal_error) and (
        base_err_vector_norm < robot.args.goal_error
    ):
        breakFlag = True

    log_item["qs"] = robot.q
    log_item["dqs"] = robot.v
    log_item["dqs_cmd"] = v_cmd.reshape((robot.model.nv,))
    log_item["ee_err_norm"] = ee_err_vector_norm.reshape((1,))
    log_item["base_err_norm"] = base_err_vector_norm.reshape((1,))
    return breakFlag, save_past_item, log_item


def DualEEAndBaseP2PCtrlLoopTemplate(
    SOLVER: Any,
    T_w_absgoal: SE3,
    T_abs_l: SE3,
    T_abs_r: SE3,
    p_basegoal: np.ndarray,
    control_loop: Callable[
        [
            Any,
            SE3,
            SE3,
            SE3,
            np.ndarray,
            Namespace,
            DualArmWholeBodyInterface,
            int,
            dict[str, deque[np.ndarray]],
        ],
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    ],
    args: Namespace,
    robot: DualArmWholeBodyInterface,
    t: int,  # will be float eventually
    past_data: dict[str, deque[np.ndarray]],
) -> control_loop_return:
    """
    DualEEAndBaseP2PCtrlLoopTemplate
    ---------------
    generic control loop for point to point motion for end-effectors of a dual arm robot
    (handling error to final point etc).
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}

    v_cmd, past_item_inner, log_item_inner = control_loop(
        SOLVER, T_w_absgoal, T_abs_l, T_abs_r, p_basegoal, args, robot, t, past_data
    )
    robot.sendVelocityCommand(v_cmd)
    log_item.update(log_item_inner)
    save_past_item.update(past_item_inner)

    T_w_lgoal = T_abs_l.act(T_w_absgoal)
    T_w_rgoal = T_abs_r.act(T_w_absgoal)
    SEerror_left = robot.T_w_l.actInv(T_w_lgoal)
    SEerror_right = robot.T_w_r.actInv(T_w_rgoal)
    err_vector_left = log6(SEerror_left).vector
    err_vector_right = log6(SEerror_right).vector
    err_vector_left_norm = np.linalg.norm(err_vector_left)
    err_vector_right_norm = np.linalg.norm(err_vector_right)

    p_base = np.array(list(robot.q[:2]) + [0.0])
    base_err_vector_norm = np.linalg.norm(p_basegoal - p_base)

    if (
        (err_vector_left_norm < robot.args.goal_error)
        and (err_vector_right_norm < robot.args.goal_error)
        and (base_err_vector_norm < robot.args.goal_error)
    ):
        breakFlag = True

    log_item["qs"] = robot.q
    log_item["dqs"] = robot.v
    log_item["dqs_cmd"] = v_cmd.reshape((robot.model.nv,))
    log_item["l_err_norm"] = err_vector_left_norm.reshape((1,))
    log_item["r_err_norm"] = err_vector_right_norm.reshape((1,))
    log_item["base_err_norm"] = base_err_vector_norm.reshape((1,))
    return breakFlag, save_past_item, log_item

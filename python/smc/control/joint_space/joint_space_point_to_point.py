from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.control.control_loop_manager import ControlLoopManager

import numpy as np
from functools import partial
from collections import deque
from argparse import Namespace
import pinocchio as pin

from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface


def moveJControlLoop(
    q_desired: np.ndarray,
    args: Namespace,
    robot: AbstractRobotManager,
    i: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    moveJControlLoop
    ---------------
    most basic P control for joint space point-to-point motion, actual control loop.
    """
    breakFlag = False
    q = robot.q
    # TODO: make a robot output a different model depending on control mode
    # to avoid having this issue
    if (robot.control_mode == AbstractRobotManager.control_mode.whole_body) or (
        robot.control_mode == AbstractRobotManager.control_mode.base_only
    ):
        q_error = pin.difference(robot.model, q, q_desired)
    else:
        if issubclass(robot.__class__, MobileBaseInterface):
            q_desired = q_desired[4:]
        q_error = q_desired - q
    # q_error = pin.difference(robot.model, q, q_desired)

    err_norm = np.linalg.norm(q_error)
    if err_norm < 1e-3:
        breakFlag = True
    K = 320
    qd = K * q_error * robot.dt
    robot.sendVelocityCommand(qd)
    log_item = {
        "qs": robot.q,
        "dqs": robot.v,
        "dqs_cmd": qd,
        "err_norm": err_norm.reshape((1,)),
    }
    return breakFlag, {}, log_item


# TODO:
# fix this by tuning or whatever else.
# MOVEL works just fine, so apply whatever's missing for there here
# and that's it.
def moveJP(q_desired: np.ndarray, args: Namespace, robot: AbstractRobotManager) -> None:
    """
    moveJP
    ---------------
    most basic P control for joint space point-to-point motion.
    just starts the control loop without any logging.
    """
    assert type(q_desired) == np.ndarray
    controlLoop = partial(moveJControlLoop, q_desired, args, robot)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros((1,)),
    }
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()
    if args.debug_prints:
        print("MoveJP done: convergence achieved, reached destionation!")


def moveJPIControlLoop(
    q_desired: np.ndarray,
    args: Namespace,
    robot: AbstractRobotManager,
    i: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    PID control for joint space point-to-point motion with approximated joint velocities.
    """

    # ================================
    # Initialization
    # ================================
    breakFlag = False
    save_past_dict = {}
    log_item = {}

    # ================================
    # Current Joint Positions
    # ================================
    q = robot.q

    # ================================
    # Compute Position Error
    # ================================
    q_error = q_desired - q  # Position error

    # ================================
    # Check for Convergence
    # ================================
    if np.linalg.norm(q_error) < 1e-3 and np.linalg.norm(robot.v) < 1e-3:
        breakFlag = True

    # ================================
    # Update Integral of Error
    # ================================
    integral_error = past_data["integral_error"][-1]
    integral_error = np.array(integral_error, dtype=np.float64).flatten()
    integral_error += q_error * robot.dt  # Accumulate error over time

    # Anti-windup: Limit integral error to prevent excessive accumulation
    max_integral = 10
    integral_error = np.clip(integral_error, -max_integral, max_integral)

    # ================================
    # Save Current States for Next Iteration
    # ================================
    save_past_dict["integral_error"] = integral_error  # Save updated integral error
    save_past_dict["q_prev"] = q  # Save current joint positions
    save_past_dict["e_prev"] = q_error  # Save current position error

    # ================================
    # Control Gains
    # ================================
    Kp = 7.0  # Proportional gain
    Ki = 0.0  # Integral gain

    # ================================
    # Compute Control Input (Joint Velocities)
    # ================================
    v_cmd = Kp * q_error + Ki * integral_error

    # ================================
    # Send Joint Velocities to the Robot
    # ================================
    robot.sendVelocityCommand(v_cmd)

    log_item["qs"] = q
    log_item["dqs"] = robot.v
    log_item["integral_error"] = integral_error.flatten()  # Save updated integral error
    log_item["e_prev"] = q_error.flatten()  # Save current position error
    return breakFlag, save_past_dict, log_item


def moveJPI(
    q_desired: np.ndarray, args: Namespace, robot: AbstractRobotManager
) -> None:
    assert isinstance(q_desired, np.ndarray)
    controlLoop = partial(moveJPIControlLoop, q_desired, args, robot)

    initial_q = robot.q
    save_past_dict = {
        "integral_error": np.zeros(robot.nq),
        "q_prev": initial_q,
        "e_prev": q_desired - initial_q,
    }

    log_item = {
        "qs": np.zeros(6),
        "dqs": np.zeros(6),
        "integral_error": np.zeros(robot.nq),
        "e_prev": q_desired - initial_q,
    }

    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()

    if args.debug_prints:
        print("MoveJPI done: convergence achieved, reached destination!")

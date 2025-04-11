from smc.control.control_loop_manager import ControlLoopManager
from smc.control.controller_templates.point_to_point import EEP2PCtrlLoopTemplate
from smc.robots.interfaces.force_torque_sensor_interface import (
    ForceTorqueOnSingleArmWrist,
)
from smc.control.cartesian_space.ik_solvers import *
from functools import partial
import pinocchio as pin
import numpy as np
import copy
from argparse import Namespace
from collections import deque


def controlLoopCompliantClik(
    #                       J           err_vec     v_cmd
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    T_w_goal: pin.SE3,
    args: Namespace,
    robot: ForceTorqueOnSingleArmWrist,
    i: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopCompliantClik
    ---------------
    CLIK with compliance in all directions
    """
    save_past_item = {}
    log_item = {}
    T_w_e = robot.T_w_e
    wrench = robot.wrench

    # we need to overcome noise if we want to converge
    if np.linalg.norm(wrench) < args.minimum_detectable_force_norm:
        wrench = np.zeros(6)
    save_past_item["wrench"] = copy.deepcopy(wrench)

    # low pass filter for wrenches
    wrench = args.beta * wrench + (1 - args.beta) * np.average(
        np.array(past_data["wrench"]), axis=0
    )
    if not args.z_only:
        Z = np.diag(np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]))
    else:
        Z = np.diag(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    wrench = Z @ wrench

    SEerror = T_w_e.actInv(T_w_goal)
    err_vector = pin.log6(SEerror).vector

    J = robot.getJacobian()
    v_cartesian_body_cmd = args.kp * err_vector + args.alpha * wrench
    v_cmd = ik_solver(J, v_cartesian_body_cmd)

    log_item["wrench"] = robot.wrench
    log_item["wrench_used"] = wrench
    return v_cmd, save_past_item, log_item


# add a threshold for the wrench
def compliantMoveL(
    T_w_goal: pin.SE3, args: Namespace, robot: ForceTorqueOnSingleArmWrist, run=True
) -> None:
    """
    compliantMoveL
    -----
    does compliantMoveL - a moveL, but with compliance achieved
    through f/t feedback
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    assert type(T_w_goal) is pin.SE3
    ik_solver = getIKSolver(args, robot)
    controlLoop = partial(
        EEP2PCtrlLoopTemplate,
        ik_solver,
        T_w_goal,
        controlLoopCompliantClik,
        args,
        robot,
    )
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "err_norm": np.zeros(1),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "wrench": np.zeros(6),
        "wrench_used": np.zeros(6),
    }
    save_past_dict = {
        "wrench": np.zeros(6),
    }
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

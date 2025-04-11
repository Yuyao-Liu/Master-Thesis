from smc.control.controller_templates.point_to_point import (
    EEAndBaseP2PCtrlLoopTemplate,
)
from smc.robots.interfaces.whole_body_single_arm_interface import (
    SingleArmWholeBodyInterface,
)
from smc.control.optimal_control.abstract_croco_ocp import CrocoOCP
from smc.control.optimal_control.croco_point_to_point.ocp.base_and_single_arm_reference_ocp import (
    BaseAndSingleArmIKOCP,
)
from smc.control.control_loop_manager import ControlLoopManager

import pinocchio as pin
import numpy as np
from functools import partial
from collections import deque
from argparse import Namespace


def CrocoP2PEEAndBaseMPCControlLoop(
    ocp: CrocoOCP,
    T_w_eegoal: pin.SE3,
    p_basegoal: np.ndarray,
    args: Namespace,
    robot: SingleArmWholeBodyInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    CrocoP2PEEAndBaseMPCControlLoop
    ---------------------
    """
    # set initial state from sensor
    x0 = np.concatenate([robot.q, robot.v])
    ocp.warmstartAndReSolve(x0)
    xs = np.array(ocp.solver.xs)
    # NOTE: for some reason the first value is always some wild bs
    vel_cmd = xs[1, robot.model.nq :]
    return vel_cmd, {}, {}


def CrocoEEAndBaseP2PMPC(
    args, robot, T_w_eegoal: pin.SE3, p_basegoal: np.ndarray, run=True
):
    """
    IKMPC
    -----
    run mpc for a point-to-point inverse kinematics.
    note that the actual problem is solved on
    a dynamics level, and velocities we command
    are actually extracted from the state x(q,dq)
    """
    x0 = np.concatenate([robot.q, robot.v])
    goal = (T_w_eegoal, p_basegoal)
    ocp = BaseAndSingleArmIKOCP(args, robot, x0, goal)
    ocp.solveInitialOCP(x0)

    controlLoop = partial(
        EEAndBaseP2PCtrlLoopTemplate,
        ocp,
        T_w_eegoal,
        p_basegoal,
        CrocoP2PEEAndBaseMPCControlLoop,
        args,
        robot,
    )
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "ee_err_norm": np.zeros(1),
        "base_err_norm": np.zeros(1),
    }
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

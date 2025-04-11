from smc.control.controller_templates.point_to_point import (
    DualEEAndBaseP2PCtrlLoopTemplate,
)
from smc.robots.interfaces.whole_body_dual_arm_interface import (
    DualArmWholeBodyInterface,
)
from smc.control.optimal_control.croco_point_to_point.ocp.base_and_dual_arm_reference_ocp import (
    BaseAndDualArmIKOCP,
)
from smc.control.control_loop_manager import ControlLoopManager

import pinocchio as pin
import numpy as np
from functools import partial
from collections import deque
from argparse import Namespace


def CrocoP2PDualEEAndBaseMPCControlLoop(
    ocp: BaseAndDualArmIKOCP,
    T_w_absgoal: pin.SE3,
    T_absgoal_l: pin.SE3,
    T_absgoal_r: pin.SE3,
    p_basegoal: np.ndarray,
    args: Namespace,
    robot: DualArmWholeBodyInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    CrocoP2PEEAndBaseMPCControlLoop
    ---------------------
    """
    x0 = np.concatenate([robot.q, robot.v])
    ocp.warmstartAndReSolve(x0)
    xs = np.array(ocp.solver.xs)
    # NOTE: for some reason the first value is always some wild bs
    v_cmd = xs[1, robot.model.nq :]

    return v_cmd, {}, {}


def CrocoDualEEAndBaseP2PMPC(
    args: Namespace,
    robot: DualArmWholeBodyInterface,
    T_w_absgoal: pin.SE3,
    T_absgoal_l: pin.SE3,
    T_absgoal_r: pin.SE3,
    p_basegoal: np.ndarray,
    run=True,
) -> None | ControlLoopManager:
    """
    IKMPC
    -----
    run mpc for a point-to-point inverse kinematics.
    note that the actual problem is solved on
    a dynamics level, and velocities we command
    are actually extracted from the state x(q,dq)
    """
    x0 = np.concatenate([robot.q, robot.v])
    T_w_lgoal = T_absgoal_l.act(T_w_absgoal)
    T_w_rgoal = T_absgoal_r.act(T_w_absgoal)
    goal = (T_w_lgoal, T_w_rgoal, p_basegoal)
    ocp = BaseAndDualArmIKOCP(args, robot, x0, goal)
    ocp.solveInitialOCP(x0)

    controlLoop = partial(
        DualEEAndBaseP2PCtrlLoopTemplate,
        ocp,
        T_w_absgoal,
        T_absgoal_l,
        T_absgoal_r,
        p_basegoal,
        CrocoP2PDualEEAndBaseMPCControlLoop,
        args,
        robot,
    )
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "l_err_norm": np.zeros(1),
        "r_err_norm": np.zeros(1),
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

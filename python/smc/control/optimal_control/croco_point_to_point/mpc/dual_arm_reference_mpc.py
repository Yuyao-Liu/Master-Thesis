from smc.control.controller_templates.point_to_point import (
    DualEEP2PCtrlLoopTemplate,
)
from smc.robots.interfaces.dual_arm_interface import DualArmInterface
from smc.control.optimal_control.abstract_croco_ocp import CrocoOCP
from smc.control.optimal_control.croco_point_to_point.ocp.dual_arm_reference_ocp import (
    DualArmIKOCP,
)
from smc.control.control_loop_manager import ControlLoopManager

import pinocchio as pin
import numpy as np
from functools import partial
from collections import deque
from argparse import Namespace

from IPython import embed


def CrocoDualEEP2PMPCControlLoop(
    ocp: CrocoOCP,
    T_w_absgoal: pin.SE3,
    T_absgoal_l: pin.SE3,
    T_absgoal_r: pin.SE3,
    args: Namespace,
    robot: DualArmInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    CrocoDualEEP2PMPCControlLoop
    ---------------------
    """
    # set initial state from sensor
    x0 = np.concatenate([robot.q, robot.v])

    T_w_lgoal = T_absgoal_l.act(T_w_absgoal)
    T_w_rgoal = T_absgoal_r.act(T_w_absgoal)
    SEerror_left = robot.T_w_l.actInv(T_w_lgoal)
    SEerror_right = robot.T_w_r.actInv(T_w_rgoal)

    # update costs depending on goal proximity
    err_vector_left = pin.log6(SEerror_left).vector
    err_vector_right = pin.log6(SEerror_right).vector
    err_vector_left_norm = np.linalg.norm(err_vector_left)
    err_vector_right_norm = np.linalg.norm(err_vector_right)

    # completely arbitrary numbers in condition
    if (err_vector_left_norm < 0.7) and (err_vector_right_norm < 0.7):
        ocp.terminalCostModel.changeCostStatus("velFinal_l", True)
        ocp.terminalCostModel.changeCostStatus("velFinal_r", True)

    ocp.warmstartAndReSolve(x0)
    xs = np.array(ocp.solver.xs)
    # NOTE: for some reason the first value is always some wild bs
    vel_cmd = xs[1, robot.model.nq :]
    return vel_cmd, {}, {}


def CrocoDualEEP2PMPC(
    args: Namespace,
    robot: DualArmInterface,
    T_w_absgoal: pin.SE3,
    T_absgoal_l: pin.SE3,
    T_absgoal_r: pin.SE3,
    run=True,
):
    """
    DualEEP2PMPC
    -----
    run mpc for a point-to-point inverse kinematics.
    note that the actual problem is solved on
    a dynamics level, and velocities we command
    are actually extracted from the state x(q,dq)
    """
    x0 = np.concatenate([robot.q, robot.v])
    T_w_lgoal = T_absgoal_l.act(T_w_absgoal)
    T_w_rgoal = T_absgoal_r.act(T_w_absgoal)

    ocp = DualArmIKOCP(args, robot, x0, (T_w_lgoal, T_w_rgoal))
    ocp.terminalCostModel.changeCostStatus("velFinal_l", False)
    ocp.terminalCostModel.changeCostStatus("velFinal_r", False)
    ocp.solveInitialOCP(x0)

    controlLoop = partial(
        DualEEP2PCtrlLoopTemplate,
        ocp,
        T_w_absgoal,
        T_absgoal_l,
        T_absgoal_r,
        CrocoDualEEP2PMPCControlLoop,
        args,
        robot,
    )
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

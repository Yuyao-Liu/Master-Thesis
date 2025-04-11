from smc.control.controller_templates.point_to_point import (
    EEP2PCtrlLoopTemplate,
)
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.control.optimal_control.abstract_croco_ocp import CrocoOCP
from smc.control.optimal_control.croco_point_to_point.ocp.single_arm_reference_ocp import (
    SingleArmIKOCP,
)
from smc.control.control_loop_manager import ControlLoopManager

import pinocchio as pin
import numpy as np
from functools import partial
from collections import deque
from argparse import Namespace


def CrocoEEP2PMPCControlLoop(
    ocp: CrocoOCP,
    T_w_goal: pin.SE3,
    args: Namespace,
    robot: SingleArmInterface,
    _: int,
    __: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    CrocoIKMPCControlLoop
    ---------------------
    """
    # set initial state from sensor
    x0 = np.concatenate([robot.q, robot.v])
    ocp.warmstartAndReSolve(x0)
    xs = np.array(ocp.solver.xs)
    # NOTE: for some reason the first value is always some wild bs
    vel_cmd = xs[1, robot.model.nq :]
    return vel_cmd, {}, {}


def CrocoEEP2PMPC(
    args: Namespace, robot: SingleArmInterface, T_w_goal: pin.SE3, run=True
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
    ocp = SingleArmIKOCP(args, robot, x0, T_w_goal)
    ocp.solveInitialOCP(x0)

    controlLoop = partial(
        EEP2PCtrlLoopTemplate, ocp, T_w_goal, CrocoEEP2PMPCControlLoop, args, robot
    )
    log_item = {
        "qs": np.zeros(robot.nq),
        "err_norm": np.zeros(1),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
    }
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

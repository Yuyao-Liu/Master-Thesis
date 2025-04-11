from smc.robots.interfaces.whole_body_single_arm_interface import (
    SingleArmWholeBodyInterface,
)
from smc.control.control_loop_manager import ControlLoopManager
from smc.control.optimal_control.abstract_croco_ocp import CrocoOCP
from smc.control.optimal_control.croco_path_following.ocp.base_and_single_arm_reference_ocp import (
    BaseAndEEPathFollowingOCP,
)
from smc.path_generation.path_math.path_to_trajectory import (
    path2D_to_trajectory2D,
    pathSE3_to_trajectorySE3,
)
from smc.control.controller_templates.path_following_template import (
    PathFollowingFromPlannerCtrllLoopTemplate,
)

import numpy as np
from functools import partial
import types
from argparse import Namespace
from pinocchio import SE3, log6
from collections import deque


def BaseAndEEPathFollowingMPCControlLoop(
    ocp: CrocoOCP,
    path: tuple[np.ndarray, list[SE3]],
    args: Namespace,
    robot,
    t: int,
    _: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    BaseAndEEPathFollowingMPCControlLoop
    -----------------------------
    both a path for both the base and the end-effector are provided,
    and both are followed
    """
    log_item = {}

    path_base, path_EE = path

    max_base_v = np.linalg.norm(robot._max_v[:2])
    perc_of_max_v = 0.5
    velocity = perc_of_max_v * max_base_v
    path_base = np.array(path_base)
    trajectory_base = path2D_to_trajectory2D(args, path_base, velocity)
    trajectory_base = np.hstack((trajectory_base, np.zeros((len(trajectory_base), 1))))
    trajectory_EE = pathSE3_to_trajectorySE3(args, path_EE, velocity)

    if t % int(np.ceil(args.ctrl_freq / 25)) == 0:
        robot.visualizer_manager.sendCommand({"path": trajectory_base})
        robot.visualizer_manager.sendCommand({"frame_path": trajectory_EE})

    x0 = np.concatenate([robot.q, robot.v])
    ocp.warmstartAndReSolve(x0, data=(trajectory_base, trajectory_EE))
    xs = np.array(ocp.solver.xs)
    v_cmd = xs[1, robot.model.nq :]

    err_vector_ee = log6(robot.T_w_e.actInv(path_EE[0]))
    err_vector_base = np.linalg.norm(
        robot.T_w_b.translation[:2] - path_base[0][:2]
    )  # z axis is irrelevant
    log_item["err_vec_ee"] = err_vector_ee
    log_item["err_norm_base"] = np.linalg.norm(err_vector_base).reshape((1,))
    return v_cmd, {}, log_item


def BaseAndEEPathFollowingMPC(
    args: Namespace,
    robot: SingleArmWholeBodyInterface,
    path_planner: types.FunctionType,
    run=True,
) -> None | ControlLoopManager:
    """
    BaseAndEEPathFollowingMPC
    -----
    run mpc for a point-to-point inverse kinematics.
    note that the actual problem is solved on
    a dynamics level, and velocities we command
    are actually extracted from the state x(q,dq).
    """

    robot._mode = SingleArmWholeBodyInterface.control_mode.whole_body
    x0 = np.concatenate([robot.q, robot.v])
    ocp = BaseAndEEPathFollowingOCP(args, robot, x0)
    ocp.solveInitialOCP(x0)

    # NOTE: for this loop it's arbitrarily decided that
    # the end-effector position is the key thing,
    # while the base is just supposed to not get in the way basically.
    # so where you are on the path is determined by the end-effector
    get_position = lambda robot: robot.T_w_e
    controlLoop = partial(
        PathFollowingFromPlannerCtrllLoopTemplate,
        path_planner,
        get_position,
        ocp,
        BaseAndEEPathFollowingMPCControlLoop,
        args,
        robot,
    )
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "err_vec_ee": np.zeros((6,)),
        "err_norm_base": np.zeros((1,)),
    }
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)

    if run:
        loop_manager.run()
    else:
        return loop_manager

from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface
from smc.control.control_loop_manager import ControlLoopManager
from smc.multiprocessing.process_manager import ProcessManager
from smc.control.optimal_control.croco_path_following.ocp.base_reference_ocp import (
    BasePathFollowingOCP,
)
from smc.control.controller_templates.path_following_template import (
    PathFollowingFromPlannerCtrllLoopTemplate,
)
from smc.path_generation.path_math.path_to_trajectory import path2D_to_trajectory2D

import numpy as np
from functools import partial
import types
from argparse import Namespace
from collections import deque


def CrocoBasePathFollowingMPCControlLoop(
    ocp: BasePathFollowingOCP,
    path2D: np.ndarray,
    args: Namespace,
    robot: MobileBaseInterface,
    t: int,
    _: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:

    p = robot.T_w_b.translation[:2]
    max_base_v = np.linalg.norm(robot._max_v[:2])
    path_base = path2D_to_trajectory2D(args, path2D, max_base_v)
    path_base = np.hstack((path_base, np.zeros((len(path_base), 1))))

    if args.visualizer:
        if t % int(np.ceil(args.ctrl_freq / 25)) == 0:
            robot.visualizer_manager.sendCommand({"path": path_base})

    x0 = np.concatenate([robot.q, robot.v])
    ocp.warmstartAndReSolve(x0, data=(path_base))
    xs = np.array(ocp.solver.xs)
    v_cmd = xs[1, robot.model.nq :]
    # NOTE: we might get stuck
    # TODO: make it more robust, it can still get stuck with this
    # a good idea is to do this for a some number of iterations -
    # you can keep this them in past data
    if np.linalg.norm(v_cmd) < 0.05:
        print(t, "RESOLVING FOR ONLY FINAL PATH POINT")
        last_point_only = np.ones((len(path_base), 2))
        last_point_only = np.hstack((last_point_only, np.zeros((len(path_base), 1))))
        last_point_only = last_point_only * path_base[-1]
        ocp.warmstartAndReSolve(x0, data=(last_point_only))
        xs = np.array(ocp.solver.xs)
        v_cmd = xs[1, robot.model.nq :]

    # NOTE: dirty hack for wholebody mode
    # v_cmd[3:] = 0.0
    err_vector_base = np.linalg.norm(p - path_base[0][:2])  # z axis is irrelevant
    log_item = {}
    log_item["err_norm_base"] = np.linalg.norm(err_vector_base).reshape((1,))
    return v_cmd, {}, log_item


def CrocoBasePathFollowingMPC(
    args: Namespace,
    robot: MobileBaseInterface,
    x0: np.ndarray,
    path_planner: ProcessManager | types.FunctionType,
    run=True,
) -> None | ControlLoopManager:
    """
    CrocoBasePathFollowingMPC
    -----
    """

    ocp = BasePathFollowingOCP(args, robot, x0)
    x0 = np.concatenate([robot.q, robot.v])
    ocp.solveInitialOCP(x0)

    get_position = lambda robot: robot.T_w_b.translation[:2]
    controlLoop = partial(
        PathFollowingFromPlannerCtrllLoopTemplate,
        path_planner,
        get_position,
        ocp,
        CrocoBasePathFollowingMPCControlLoop,
        args,
        robot,
    )
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "err_norm_base": np.zeros((1,)),
    }
    save_past_item = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_item, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.control.optimal_control.croco_path_following.ocp.single_arm_reference_ocp import (
    CrocoEEPathFollowingOCP,
)
from smc.control.controller_templates.path_following_template import (
    PathFollowingFromPlannerCtrllLoopTemplate,
)
from smc.path_generation.path_math.path2d_to_6d import (
    path2D_to_SE3,
)
from smc.path_generation.path_math.path_to_trajectory import (
    path2D_to_trajectory2D,
    pathSE3_to_trajectorySE3,
)
from smc.control.control_loop_manager import ControlLoopManager
from smc.multiprocessing.process_manager import ProcessManager

import numpy as np
from functools import partial
import types
from argparse import Namespace
from pinocchio import SE3, log6
from collections import deque


def CrocoEEPathFollowingMPCControlLoop(
    ocp: CrocoEEPathFollowingOCP,
    path: np.ndarray | list[SE3],  # either 2D or pose
    args: Namespace,
    robot: SingleArmInterface,
    t: int,
    _: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    CrocoPathFollowingMPCControlLoop
    -----------------------------
    end-effector follows the prescribed path.

    path planner can either be a function which spits out a list of path points
    or an instance of ProcessManager which spits out path points
    by calling ProcessManager.getData()
    """
    max_base_v = np.linalg.norm(robot._max_v[:2])
    perc_of_max_v = 0.5
    velocity = perc_of_max_v * max_base_v
    if type(path) == np.ndarray:
        trajectory2D = path2D_to_trajectory2D(args, path, velocity)
        trajectory_EE_SE3 = path2D_to_SE3(trajectory2D, args.handlebar_height)
    else:
        trajectory_EE_SE3 = pathSE3_to_trajectorySE3(args, path, velocity)

    # TODO: EVIL AND HAS TO BE REMOVED FROM HERE
    if args.visualizer:
        if t % int(np.ceil(args.ctrl_freq / 25)) == 0:
            robot.visualizer_manager.sendCommand({"frame_path": trajectory_EE_SE3})

    x0 = np.concatenate([robot.q, robot.v])
    ocp.warmstartAndReSolve(x0, data=trajectory_EE_SE3)
    xs = np.array(ocp.solver.xs)
    v_cmd = xs[1, robot.model.nq :]

    err_vector_ee = log6(robot.T_w_e.actInv(trajectory_EE_SE3[0])).vector
    log_item = {"err_vec_ee": err_vector_ee}

    return v_cmd, {}, log_item


def CrocoEEPathFollowingMPC(
    args: Namespace,
    robot: SingleArmInterface,
    x0: np.ndarray,
    path_planner: ProcessManager | types.FunctionType,
    run=True,
) -> None | ControlLoopManager:
    """
    CrocoEndEffectorPathFollowingMPC
    -----
    follow a fixed pre-determined path, or a path received from a planner.
    the path does NOT need to start from your current pose - you need to get to it yourself.
    """

    ocp = CrocoEEPathFollowingOCP(args, robot, x0)
    # technically should be done in controlloop because now
    # it's solved 2 times before the first command,
    # but we don't have time for details rn
    x0 = np.concatenate([robot.q, robot.v])
    ocp.solveInitialOCP(x0)

    # NOTE: a bit of dirty hacking for now because
    # i know that the only planner i have gives 2D references.
    # but of course otherwise the only reasonable
    # thing is that the planner gives an SE3 reference
    # to a MPC following an SE3 reference
    if type(path_planner) == ProcessManager:
        get_position = lambda robot: robot.T_w_e.translation[:2]
    else:
        get_position = lambda robot: robot.T_w_e
    controlLoop = partial(
        PathFollowingFromPlannerCtrllLoopTemplate,
        path_planner,
        get_position,
        ocp,
        CrocoEEPathFollowingMPCControlLoop,
        args,
        robot,
    )
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "err_vec_ee": np.zeros(6),
    }
    save_past_item = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_item, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

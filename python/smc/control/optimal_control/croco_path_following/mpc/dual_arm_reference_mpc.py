from smc.robots.interfaces.dual_arm_interface import DualArmInterface
from smc.control.optimal_control.croco_path_following.ocp.dual_arm_reference_ocp import (
    DualArmEEPathFollowingOCP,
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


def CrocoDualArmEEPathFollowingMPCControlLoop(
    T_absgoal_l: SE3,
    T_absgoal_r: SE3,
    ocp: DualArmEEPathFollowingOCP,
    path: np.ndarray | list[SE3],  # either 2D or pose
    args: Namespace,
    robot: DualArmInterface,
    t: int,
    _: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    CrocoDualArmEEPathFollowingMPCControlLoop
    -----------------------------
    end-effectors follows the prescribed path.
    the path is defined with T_w_abs, from which
    T_w_l and T_w_r are defined via T_absgoal_l and T_absgoal_r.

    path planner can either be a function which spits out a list of path points
    or an instance of ProcessManager which spits out path points
    by calling ProcessManager.getData()
    """
    max_base_v = np.linalg.norm(robot._max_v[:2])
    perc_of_max_v = 0.5
    velocity = perc_of_max_v * max_base_v
    if type(path) == np.ndarray:
        trajectory2D = path2D_to_trajectory2D(args, path, velocity)
        trajectory_T_w_abs = path2D_to_SE3(trajectory2D, args.handlebar_height)
    else:
        trajectory_T_w_abs = pathSE3_to_trajectorySE3(args, path, velocity)

    trajectory_T_w_l = []
    trajectory_T_w_r = []
    for traj_T_w_abs in trajectory_T_w_abs:
        trajectory_T_w_l.append(T_absgoal_l.act(traj_T_w_abs))
        trajectory_T_w_r.append(T_absgoal_r.act(traj_T_w_abs))

    # TODO: EVIL AND HAS TO BE REMOVED FROM HERE
    if args.visualizer:
        if t % int(np.ceil(args.ctrl_freq / 25)) == 0:
            robot.visualizer_manager.sendCommand({"frame_path": trajectory_T_w_abs})

    x0 = np.concatenate([robot.q, robot.v])
    ocp.warmstartAndReSolve(x0, data=(trajectory_T_w_l, trajectory_T_w_r))
    xs = np.array(ocp.solver.xs)
    v_cmd = xs[1, robot.model.nq :]

    err_vector_ee_l = log6(robot.T_w_l.actInv(trajectory_T_w_l[0])).vector
    err_vector_ee_r = log6(robot.T_w_r.actInv(trajectory_T_w_r[0])).vector
    log_item = {"err_vec_ee_l": err_vector_ee_l, "err_vec_ee_r": err_vector_ee_r}

    return v_cmd, {}, log_item


def CrocoDualArmEEPathFollowingMPC(
    args: Namespace,
    robot: DualArmInterface,
    T_absgoal_l: SE3,
    T_absgoal_r: SE3,
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

    ocp = DualArmEEPathFollowingOCP(args, robot, x0)
    # ocp.terminalCostModel.changeCostStatus("velFinal_l", False)
    # ocp.terminalCostModel.changeCostStatus("velFinal_r", False)
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
        get_position = lambda robot: robot.T_w_abs.translation[:2]
    else:
        get_position = lambda robot: robot.T_w_abs
    loop = partial(CrocoDualArmEEPathFollowingMPCControlLoop, T_absgoal_l, T_absgoal_r)
    controlLoop = partial(
        PathFollowingFromPlannerCtrllLoopTemplate,
        path_planner,
        get_position,
        ocp,
        loop,
        args,
        robot,
    )
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "err_vec_ee_l": np.zeros(6),
        "err_vec_ee_r": np.zeros(6),
    }
    save_past_item = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_item, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

from smc.control.control_loop_manager import ControlLoopManager
from smc.multiprocessing.process_manager import ProcessManager
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.control.controller_templates.path_following_template import (
    PathFollowingFromPlannerCtrllLoopTemplate,
)
from smc.control.cartesian_space.ik_solvers import getIKSolver, dampedPseudoinverse
from smc.path_generation.path_math.path2d_to_6d import (
    path2D_to_SE3,
)

from functools import partial
import pinocchio as pin
import numpy as np
from argparse import Namespace
from collections import deque
from typing import Callable
import types


def cartesianPathFollowingControlLoop(
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    path: list[pin.SE3] | np.ndarray,
    args: Namespace,
    robot: SingleArmInterface,
    t: int,
    _: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    cartesianPathFollowingControlLoop
    -----------------------------
    end-effector(s) follow their path(s) according to what a 2D path-planner spits out
    """

    # TODO: refactor this horror out of here
    if type(path) == np.ndarray:
        path = path2D_to_SE3(path[:, :2], 0.0)
    # TODO: arbitrary bs, read a book and redo this
    # NOTE: assuming the first path point coincides with current pose
    SEerror = robot.T_w_e.actInv(path[1])
    V_path = pin.log6(path[0].actInv(path[1])).vector
    err_vector = pin.log6(SEerror).vector + V_path
    J = robot.getJacobian()
    v_cmd = ik_solver(J, err_vector)

    if v_cmd is None:
        print(
            t,
            "the controller you chose produced None as output, using dampedPseudoinverse instead",
        )
        v_cmd = dampedPseudoinverse(1e-2, J, err_vector)
    else:
        if args.debug_prints:
            print(t, "ik solver success")

    # maybe visualize the closest path point instead? the path should be handled
    # by the path planner
    if args.visualizer:
        if t % int(np.ceil(args.ctrl_freq / 25)) == 0:
            robot.visualizer_manager.sendCommand({"frame_path": path[:20]})

    return v_cmd, {}, {}


def cartesianPathFollowingWithPlanner(
    args: Namespace,
    robot: SingleArmInterface,
    path_planner: ProcessManager | types.FunctionType,
    run=True,
) -> None | ControlLoopManager:
    ik_solver = getIKSolver(args, robot)
    get_position = lambda robot: robot.T_w_e.translation[:2]
    controlLoop = partial(
        PathFollowingFromPlannerCtrllLoopTemplate,
        path_planner,
        get_position,
        ik_solver,
        cartesianPathFollowingControlLoop,
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

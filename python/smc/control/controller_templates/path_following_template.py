from types import FunctionType
from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.multiprocessing.process_manager import ProcessManager

from argparse import Namespace
from typing import Any, Callable
import numpy as np
from collections import deque
from pinocchio import SE3

global control_loop_return
control_loop_return = tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]


def PathFollowingFromPlannerCtrllLoopTemplate(
    path_planner: ProcessManager | FunctionType,
    get_position: Callable[[AbstractRobotManager], np.ndarray],
    SOLVER: Any,
    control_loop: Callable[
        [
            Any,
            np.ndarray | list[SE3],
            Namespace,
            AbstractRobotManager,
            int,
            dict[str, deque[np.ndarray]],
        ],
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    ],
    args: Namespace,
    robot: AbstractRobotManager,
    t: int,  # will be float eventually
    past_data: dict[str, deque[np.ndarray]],
) -> control_loop_return:
    """
    PathFollowingFromPlannerControlLoop
    ---------------
    handles getting path and with comm with the planner.
    then you do whatever you want with the path, and execute a path following controller
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}

    p = get_position(robot)

    if type(path_planner) == ProcessManager:
        path_planner.sendCommand(p)
        data = path_planner.getData()

        if data == "done":
            breakFlag = True

        if data == "done" or data is None:
            robot.sendVelocityCommand(np.zeros(robot.model.nv))
            log_item["qs"] = robot.q
            log_item["dqs"] = robot.v
            return breakFlag, save_past_item, log_item

        _, path2D = data
        path2D = np.array(path2D).reshape((-1, 2))
    else:
        # NOTE: DOES NOT HAVE TO BE 2D IN THIS CASE,
        # TODO: RENAME APROPRIATELY
        path2D = path_planner(p)
        # NOTE: more evil in case path2D is a tuple.
        # i have no time to properly rewrite this so pls bear w/ me
        if len(path2D) < 4 and len(path2D[0]) < 4:
            breakFlag = True

    v_cmd, past_item_inner, log_item_inner = control_loop(
        SOLVER, path2D, args, robot, t, past_data
    )
    robot.sendVelocityCommand(v_cmd)
    log_item.update(log_item_inner)
    save_past_item.update(past_item_inner)

    log_item["qs"] = robot.q
    log_item["dqs"] = robot.v
    return breakFlag, save_past_item, log_item

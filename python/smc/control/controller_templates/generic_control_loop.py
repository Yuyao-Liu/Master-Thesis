from smc.robots.abstract_robotmanager import AbstractRobotManager

from argparse import Namespace
from typing import Any, Callable
import numpy as np
from collections import deque

global control_loop_return
control_loop_return = tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]
global inner_loop_return
inner_loop_return = tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]


def GenericControlLoop(
    X: Any,
    control_loop: Callable[
        [Any, Namespace, AbstractRobotManager, int, dict[str, deque[np.ndarray]]],
        control_loop_return,
    ],
    args: Namespace,
    robot: AbstractRobotManager,
    t: int,  # will be float eventually
    past_data: dict[str, deque[np.ndarray]],
) -> control_loop_return:
    breakFlag = False
    log_item = {}
    save_past_item = {}

    v_cmd, past_item_inner, log_item_inner = control_loop(X, args, robot, t, past_data)

    robot.sendVelocityCommand(v_cmd)

    log_item.update(log_item_inner)
    save_past_item.update(past_item_inner)
    return breakFlag, save_past_item, log_item

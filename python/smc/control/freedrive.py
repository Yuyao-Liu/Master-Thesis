from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.control.control_loop_manager import ControlLoopManager

import numpy as np
from functools import partial
import threading
from queue import Queue
from argparse import Namespace
from collections import deque
from pinocchio import SE3


def freedriveControlLoop(
    args: Namespace,
    robot: SingleArmInterface,
    com_queue: Queue,
    pose_n_q_dict: dict[str, list],
    i: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopFreedrive
    -----------------------------
    while in freedrive, collect qs.
    this can be used to visualize and plot while in freedrive,
    collect some points or a trajectory etc.
    this function does not have those features,
    but you can use this function as template to make them
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.q
    T_w_e = robot.T_w_e

    if not com_queue.empty():
        msg = com_queue.get()
        if msg == "q":
            breakFlag = True
        if msg == "s":
            pose_n_q_dict["T_w_es"].append(T_w_e.copy())
            pose_n_q_dict["qs"].append(q.copy())

    if args.debug_prints:
        print("===========================================")
        print(T_w_e)
        print("q:", *np.array(q).round(4))

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.v.reshape((robot.model.nv,))
    return breakFlag, save_past_dict, log_item


def freedriveUntilKeyboard(
    args: Namespace, robot: SingleArmInterface
) -> dict[str, list[SE3] | np.ndarray]:
    """
    controlLoopFreedrive
    -----------------------------
    while in freedrive, collect qs.
    this can be used to visualize and plot while in freedrive,
    collect some points or a trajectory etc.
    you can save the log from this function and use that,
    or type on the keyboard to save specific points (q + T_w_e)
    """
    if not args.real:
        print(
            """
    ideally now you would use some sliders or something, 
    but i don't have time to implement that. just run some movement 
    to get it where you want pls. freedrive will just exit now
            """
        )
        return {}
    robot.setFreedrive()
    # set up freedrive controlloop (does nothing, just accesses
    # all controlLoopManager goodies)
    log_item = {"qs": np.zeros((robot.model.nq,)), "dqs": np.zeros((robot.model.nv,))}
    save_past_dict = {}
    # use Queue because we're doing this on a
    # threading level, not multiprocess level
    # infinite size (won't need more than 1 but who cares)
    com_queue = Queue()
    # we're passing pose_n_q_list by reference
    # (python default for mutables)
    pose_n_q_dict = {"T_w_es": [], "qs": []}
    controlLoop = ControlLoopManager(
        robot,
        partial(freedriveControlLoop, args, robot, com_queue, pose_n_q_dict),
        args,
        save_past_dict,
        log_item,
    )

    # wait for keyboard input in a different thread
    # (obviously necessary because otherwise literally nothing else
    #  can happen)
    def waitKeyboardFunction(com_queue):
        cmd = ""
        # empty string is cast to false
        while True:
            cmd = input("Press q to stop and exit, s to save joint angle and T_w_e: ")
            if (cmd != "q") and (cmd != "s"):
                print("invalid input, only s or q (then Enter) allowed")
            else:
                com_queue.put(cmd)
                if cmd == "q":
                    break

    # we definitely want a thread and leverage GIL,
    # because the thread is literally just something to sit
    # on a blocking call from keyboard input
    # (it would almost certainly be the same without the GIL,
    #  but would maybe require stdin sharing or something)
    waitKeyboardThread = threading.Thread(
        target=waitKeyboardFunction, args=(com_queue,)
    )
    waitKeyboardThread.start()
    controlLoop.run()
    waitKeyboardThread.join()
    robot.unSetFreedrive()
    return pose_n_q_dict

import pinocchio as pin
import numpy as np
from functools import partial
from ur_simple_control.networking.client import client
from ur_simple_control.managers import (
    ProcessManager,
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
)


def get_args():
    parser = getMinimalArgParser()
    parser.description = "the robot will received joint angles from a socket and go to them in joint space"
    # add more arguments here from different Simple Manipulator Control modules
    parser.add_argument("--host", type=str, help="host ip address", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="host's port", default=7777)
    args = parser.parse_args()
    return args


def controlLoopExternalQ(robot: RobotManager, receiver: ProcessManager, i, past_data):
    """
    controlLoop
    -----------------------------
    controller description
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.getQ()

    q_desired = receiver.getData()["q"]
    q_error = q_desired - q
    K = 10
    qd_cmd = K * q_error

    robot.sendQd(qd_cmd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["q_error"] = q_error.reshape((robot.model.nq,))
    return breakFlag, save_past_dict, log_item


if __name__ == "__main__":
    args = get_args()
    robot = RobotManager(args)

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    # VERY important that the first q we'll pass as desired is the current q, meaning the robot won't move
    # this is set with init_value
    receiver = ProcessManager(
        args, client, {"q": robot.q.copy()}, 4, init_value={"q": robot.q.copy()}
    )
    log_item = {
        "qs": np.zeros((robot.model.nq,)),
        "dqs": np.zeros((robot.model.nv,)),
        "q_error": np.zeros((robot.model.nq,)),
    }
    control_loop = partial(controlLoopExternalQ, robot, receiver)
    loop_manager = ControlLoopManager(robot, control_loop, args, {}, log_item)
    loop_manager.run()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.saveLog()
    # loop_manager.stopHandler(None, None)

from ur_simple_control.vision.vision import processCamera
from ur_simple_control.managers import (
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
    ProcessManager,
)
import argcomplete, argparse
import numpy as np
import time
import pinocchio as pin
from functools import partial


def get_args():
    parser = getMinimalArgParser()
    parser.description = "dummy camera output, but being processed \
                          in a different process"
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


def controlLoopWithCamera(
    camera_manager: ProcessManager, args, robot: RobotManager, i, past_data
):
    """
    controlLoopWithCamera
    -----------------------------
    do nothing while getting dummy camera input
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}
    q = robot.getQ()

    camera_output = camera_manager.getData()
    # print(camera_output)

    qd_cmd = np.zeros(robot.model.nv)

    robot.sendQd(qd_cmd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["camera_output"] = np.array([camera_output["x"], camera_output["y"]])
    return breakFlag, save_past_dict, log_item


if __name__ == "__main__":
    args = get_args()
    robot = RobotManager(args)

    # cv2 camera device 0
    device = 0
    side_function = partial(processCamera, device)
    init_value = {"x": np.random.randint(0, 10), "y": np.random.randint(0, 10)}
    camera_manager = ProcessManager(args, side_function, {}, 1, init_value=init_value)

    log_item = {}
    log_item["qs"] = np.zeros((robot.model.nq,))
    log_item["dqs"] = np.zeros((robot.model.nv,))
    log_item["camera_output"] = np.zeros(2)
    controlLoop = partial(controlLoopWithCamera, camera_manager, args, robot)
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()

    camera_manager.terminateProcess()

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.saveLog()
        robot.log_manager.plotAllControlLoops()

import pinocchio as pin
import numpy as np
from functools import partial
from ur_simple_control.networking.client import client
from ur_simple_control.networking.server import server
from ur_simple_control.clik.clik import *
from ur_simple_control.managers import (
    ProcessManager,
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
)


def get_args():
    parser = getMinimalArgParser()
    parser = getClikArgs(parser)
    parser.description = "the robot will received joint angles from a socket and go to them in joint space"
    # add more arguments here from different Simple Manipulator Control modules
    parser.add_argument("--host", type=str, help="host ip address", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="host's port", default=7777)
    args = parser.parse_args()
    return args


def controlLoopClikExternalGoal(
    robot: RobotManager,
    receiver: ProcessManager,
    sender: ProcessManager,
    clik_controller,
    i,
    past_data,
):
    """
    controlLoop
    -----------------------------
    controller description
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    data = receiver.getData()
    T_goal = data["T_goal"]
    #print("translation:", T_goal.translation)
    #print("rotation:", pin.Quaternion(T_goal.rotation))
    SEerror = T_w_e.actInv(T_goal)
    err_vector = pin.log6(SEerror).vector
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)

    qd_cmd = (
        J.T
        @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * args.tikhonov_damp)
        @ (err_vector + data["v"])
    )
    if qd_cmd is None:
        print("the controller you chose didn't work, using dampedPseudoinverse instead")
        qd_cmd = dampedPseudoinverse(1e-2, J, err_vector)

    # NOTE: this will also receive a velocity
    # i have no idea what to do with that velocity
    robot.sendQd(qd_cmd)
    # NOTE: this one is in the base frame! (as it should be)
    # it is saved in robotmanager (because it's a private variable,
    # yes this is evil bla bla)
    wrench_in_robot = robot.getWrench()
    robot._getWrench()
    wrench = robot.getWrench()
    robot.wrench = wrench_in_robot
    sender.sendCommand({"wrench": wrench})

    log_item["qs"] = robot.getQ().reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["err_vector"] = err_vector
    return breakFlag, save_past_dict, log_item


if __name__ == "__main__":
    args = get_args()
    robot = RobotManager(args)

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    # VERY important that the first q we'll pass as desired is the current q, meaning the robot won't move
    # this is set with init_value
    # command_sender: 7777
    receiver = ProcessManager(
        args,
        client,
        {"T_goal": pin.SE3.Identity(), "v": np.zeros(6)},
        4,
        init_value={"T_goal": robot.getT_w_e(), "v": np.zeros(6)},
    )
    # wrench_sender: 6666
    args.port = 6666
    sender = ProcessManager(args, server, {"wrench": np.zeros(6)}, 0)
    log_item = {
        "qs": np.zeros((robot.model.nq,)),
        "dqs": np.zeros((robot.model.nv,)),
        "err_vector": np.zeros((6,)),
    }
    clik_controller = getClikController(args, robot)
    control_loop = partial(
        controlLoopClikExternalGoal, robot, receiver, sender, clik_controller
    )
    loop_manager = ControlLoopManager(robot, control_loop, args, {}, log_item)
    loop_manager.run()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.saveLog()
    # loop_manager.stopHandler(None, None)

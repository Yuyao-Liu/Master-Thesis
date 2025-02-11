import pinocchio as pin
import numpy as np
from functools import partial
from ur_simple_control.managers import (
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
)


def get_args():
    parser = getMinimalArgParser()
    parser.description = "force control example"
    # add more arguments here from different Simple Manipulator Control modules
    args = parser.parse_args()
    return args


def controlLoopForceEx(args, robot: RobotManager, T_w_e_init, i, past_data):
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
    wrench = robot.getWrench()
    save_past_dict["wrench"] = wrench.copy()
    wrench = 0.5 * wrench + (1 - 0.5) * np.average(
        np.array(past_data["wrench"]), axis=0
    )
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)

    Kp = 1.0
    wrench_ref = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    error = T_w_e.actInv(T_w_e_init)
    error_v = pin.log6(error).vector
    error_v[2] = 0.0
    qd_cmd = Kp * J.T @ error_v + J.T @ (wrench_ref - wrench)

    robot.sendQd(qd_cmd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["wrench"] = wrench
    return breakFlag, save_past_dict, log_item


if __name__ == "__main__":
    args = get_args()
    robot = RobotManager(args)
    q_init = robot.getQ()
    T_w_e_init = robot.getT_w_e()
    controlLoop = partial(controlLoopForceEx, args, robot, T_w_e_init)
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "wrench": np.zeros(6),
    }
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, {"wrench": np.zeros(6)}, log_item
    )
    loop_manager.run()

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.saveLog()
    # loop_manager.stopHandler(None, None)

# PYTHON_ARGCOMPLETE_OK
import pinocchio as pin
import numpy as np
import copy
import argparse, argcomplete
import time
from functools import partial
from ur_simple_control.visualize.visualize import plotFromDict
from ur_simple_control.util.draw_path import drawPath
from ur_simple_control.dmp.dmp import DMP, NoTC, TCVelAccConstrained
from ur_simple_control.clik.clik import (
    getClikArgs,
    getClikController,
    moveL,
    moveUntilContact,
)
from ur_simple_control.managers import (
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
)
from ur_simple_control.basics.basics import moveJPI


def getArgs():
    parser = getMinimalArgParser()
    parser = getClikArgs(parser)
    parser.add_argument(
        "--kp",
        type=float,
        help="proportial control constant for position errors",
        default=1.0,
    )
    parser.add_argument(
        "--kv", type=float, help="damping in impedance control", default=0.001
    )
    parser.add_argument(
        "--cartesian-space-impedance",
        action=argparse.BooleanOptionalAction,
        help="is the impedance computed and added in cartesian or in joint space",
        default=False,
    )
    parser.add_argument(
        "--z-only",
        action=argparse.BooleanOptionalAction,
        help="whether you have general impedance or just ee z axis",
        default=False,
    )

    #    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


# feedforward velocity, feedback position and force for impedance
def controller():
    pass


# control loop to be passed to ControlLoopManager
def controlLoopPointImpedance(
    args, q_init, controller, robot: RobotManager, i, past_data
):
    breakFlag = False
    # TODO rename this into something less confusing
    save_past_dict = {}
    log_item = {}
    q = robot.getQ()
    Mtool = robot.getT_w_e()
    wrench = robot.getWrench()
    log_item["wrench_raw"] = wrench.reshape((6,))
    # deepcopy for good coding practise (and correctness here)
    save_past_dict["wrench"] = copy.deepcopy(wrench)
    # rolling average
    # wrench = np.average(np.array(past_data['wrench']), axis=0)
    # first-order low pass filtering instead
    # beta is a smoothing coefficient, smaller values smooth more, has to be in [0,1]
    # wrench = args.beta * wrench + (1 - args.beta) * past_data['wrench'][-1]
    wrench = args.beta * wrench + (1 - args.beta) * np.average(
        np.array(past_data["wrench"]), axis=0
    )
    if not args.z_only:
        Z = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    else:
        Z = np.diag(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    # Z = np.diag(np.ones(6))

    wrench = Z @ wrench

    # this jacobian might be wrong
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    dq = robot.getQd()[:6].reshape((6, 1))
    # get joint
    tau = J.T @ wrench
    #    if i % 25 == 0:
    #        print(*tau.round(1))
    tau = tau[:6].reshape((6, 1))
    # compute control law:
    # - feedback the position
    # kv is not needed if we're running velocity control
    vel_cmd = (
        args.kp * (q_init[:6].reshape((6, 1)) - q[:6].reshape((6, 1)))
        + args.alpha * tau
    )
    # vel_cmd = np.zeros(6)
    robot.sendQd(vel_cmd)

    # immediatelly stop if something weird happened (some non-convergence)
    if np.isnan(vel_cmd[0]):
        breakFlag = True

    # log what you said you'd log
    # TODO fix the q6 situation (hide this)
    log_item["qs"] = q[:6].reshape((6,))
    log_item["dqs"] = dq.reshape((6,))
    log_item["wrench_used"] = wrench.reshape((6,))

    return breakFlag, save_past_dict, log_item


def controlLoopCartesianPointImpedance(
    args, Mtool_init, clik_controller, robot: RobotManager, i, past_data
):
    breakFlag = False
    # TODO rename this into something less confusing
    save_past_dict = {}
    log_item = {}
    q = robot.getQ()
    Mtool = robot.getT_w_e()
    wrench = robot.getWrench()
    log_item["wrench_raw"] = wrench.reshape((6,))
    save_past_dict["wrench"] = copy.deepcopy(wrench)
    # wrench = args.beta * wrench + (1 - args.beta) * past_data['wrench'][-1]
    wrench = args.beta * wrench + (1 - args.beta) * np.average(
        np.array(past_data["wrench"]), axis=0
    )
    # good generic values
    # Z = np.diag(np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0]))
    # but let's stick to the default for now
    if not args.z_only:
        Z = np.diag(np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]))
    else:
        Z = np.diag(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    # Z = np.diag(np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]))
    # Z = np.diag(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

    wrench = Z @ wrench

    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)

    SEerror = Mtool.actInv(Mtool_init)
    err_vector = pin.log6(SEerror).vector
    v_cartesian_body_cmd = args.kp * err_vector + args.alpha * wrench

    vel_cmd = clik_controller(J, v_cartesian_body_cmd)
    robot.sendQd(vel_cmd)

    # immediatelly stop if something weird happened (some non-convergence)
    if np.isnan(vel_cmd[0]):
        breakFlag = True

    dq = robot.getQd()[:6].reshape((6, 1))
    # log what you said you'd log
    # TODO fix the q6 situation (hide this)
    log_item["qs"] = q[:6].reshape((6,))
    log_item["dqs"] = dq.reshape((6,))
    log_item["wrench_used"] = wrench.reshape((6,))

    return breakFlag, save_past_dict, log_item


if __name__ == "__main__":
    args = getArgs()
    robot = RobotManager(args)
    clikController = getClikController(args, robot)

    # TODO and NOTE the weight, TCP and inertial matrix needs to be set on the robot
    # you already found an API in rtde_control for this, just put it in initialization
    # under using/not-using gripper parameters
    # ALSO NOTE: to use this you need to change the version inclusions in
    # ur_rtde due to a bug there in the current ur_rtde + robot firmware version
    # (the bug is it works with the firmware verion, but ur_rtde thinks it doesn't)
    # here you give what you're saving in the rolling past window
    # it's initial value.
    # controlLoopManager will populate the queue with these initial values
    save_past_dict = {
        "wrench": np.zeros(6),
    }
    # here you give it it's initial value
    log_item = {
        "qs": np.zeros(robot.n_arm_joints),
        "dqs": np.zeros(robot.n_arm_joints),
        "wrench_raw": np.zeros(6),
        "wrench_used": np.zeros(6),
    }
    q_init = robot.getQ()
    Mtool_init = robot.getT_w_e()

    if not args.cartesian_space_impedance:
        controlLoop = partial(
            controlLoopPointImpedance, args, q_init, controller, robot
        )
    else:
        controlLoop = partial(
            controlLoopCartesianPointImpedance, args, Mtool_init, clikController, robot
        )

    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()

    if not args.pinocchio_only:
        robot.stopRobot()

    if args.save_log:
        robot.log_manager.saveLog()
        robot.log_manager.plotAllControlLoops()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()

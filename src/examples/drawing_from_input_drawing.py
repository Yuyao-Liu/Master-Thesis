# PYTHON_ARGCOMPLETE_OK
# TODO: make 1/beta  = 5
import pinocchio as pin
import numpy as np
import matplotlib
import argparse, argcomplete
import time
import pickle
from functools import partial
from ur_simple_control.visualize.visualize import plotFromDict
from ur_simple_control.util.draw_path import drawPath
from ur_simple_control.dmp.dmp import (
    getDMPArgs,
    DMP,
    NoTC,
    TCVelAccConstrained,
    followDMP,
)
from ur_simple_control.clik.clik import (
    getClikArgs,
    getClikController,
    moveL,
    moveUntilContact,
    controlLoopClik,
    compliantMoveL,
    clikCartesianPathIntoJointPath,
)
from ur_simple_control.util.map2DPathTo3DPlane import map2DPathTo3DPlane
from ur_simple_control.managers import (
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
)
from ur_simple_control.util.calib_board_hacks import (
    getBoardCalibrationArgs,
    calibratePlane,
)
from ur_simple_control.basics.basics import moveJPI

#######################################################################
#                            arguments                                #
#######################################################################


def getArgs():
    parser = getMinimalArgParser()
    parser = getClikArgs(parser)
    parser = getDMPArgs(parser)
    parser = getBoardCalibrationArgs(parser)
    parser.description = "Make a drawing on screen,\
            watch the robot do it on the whiteboard."
    parser.add_argument(
        "--kp",
        type=float,
        help="proportial control constant for position errors",
        default=1.0,
    )
    parser.add_argument(
        "--mm-into-board",
        type=float,
        help="number of milimiters the path is into the board",
        default=3.0,
    )
    parser.add_argument(
        "--kv", type=float, help="damping in impedance control", default=0.001
    )
    parser.add_argument(
        "--draw-new",
        action=argparse.BooleanOptionalAction,
        help="whether draw a new picture, or use the saved path path_in_pixels.csv",
        default=True,
    )
    parser.add_argument(
        "--pick-up-marker",
        action=argparse.BooleanOptionalAction,
        help="""
    whether the robot should pick up the marker.
    NOTE: THIS IS FROM A PREDEFINED LOCATION.
    """,
        default=False,
    )
    parser.add_argument(
        "--find-marker-offset",
        action=argparse.BooleanOptionalAction,
        help="""
    whether you want to do find marker offset (recalculate TCP
    based on the marker""",
        default=True,
    )
    parser.add_argument(
        "--board-wiping",
        action=argparse.BooleanOptionalAction,
        help="are you wiping the board (default is no because you're writing)",
        default=False,
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


def getMarker(args, robot, plane_pose):
    """
    getMarker
    ---------
    get marker from user in a blind handover
    RUN THIS IN SIM FIRST TO SEE IF IT MAKES SENSE.
    if not, generate a different joint trajectory for your situation.
    """
    # load traj
    file = open("./data/from_writing_to_handover.pickle_save", "rb")
    point_dict = pickle.load(file)
    file.close()
    #####################
    #  go toward user   #
    #####################
    # this is more than enough, and it will be the same for both
    tau0 = 5
    # setting up array for dmp
    # TODO: make the dmp general already
    for i in range(len(point_dict["qs"])):
        point_dict["qs"][i] = point_dict["qs"][i][:6]
    qs = np.array(point_dict["qs"])

    followDMP(args, robot, qs, tau0)
    robot.sendQd(np.zeros(robot.model.nq))

    ##########################################
    #  blind handover (open/close on timer)  #
    ##########################################
    robot.openGripper()
    time.sleep(5)
    robot.closeGripper()
    time.sleep(3)

    #############
    #  go back  #
    #############
    point_dict["qs"].reverse()
    # setting up array for dmp
    qs = np.array(point_dict["qs"])
    followDMP(args, robot, qs, tau0)


def findMarkerOffset(args, robot: RobotManager, plane_pose: pin.SE3):
    """
    findMarkerOffset
    ---------------
    This relies on having the correct orientation of the plane
    and the correct translation vector for top-left corner.
    Idea is you pick up the marker, go to the top corner,
    touch it, and see the difference between that and the translation vector.
    Obviously it's just a hacked solution, but it works so who cares.
    """
    above_starting_write_point = pin.SE3.Identity()
    above_starting_write_point.translation[2] = -0.2
    above_starting_write_point = plane_pose.act(above_starting_write_point)
    print("going to above plane pose point", above_starting_write_point)
    compliantMoveL(args, robot, above_starting_write_point)

    # this is in the end-effector frame, so this means going straight down
    # because we are using the body jacobians in our clik
    speed = np.zeros(6)
    speed[2] = 0.02
    moveUntilContact(args, robot, speed)
    # we use the pin coordinate system because that's what's
    # the correct thing long term accross different robots etc
    current_translation = robot.getT_w_e().translation
    # i only care about the z because i'm fixing the path atm
    # but, let's account for the possible milimiter offset 'cos why not
    marker_offset = np.linalg.norm(plane_pose.translation - current_translation)

    print("going back")
    compliantMoveL(args, robot, above_starting_write_point)
    return marker_offset


def controlLoopWriting(args, robot: RobotManager, dmp, tc, i, past_data):
    """
    controlLoopWriting
    -----------------------
    dmp reference on joint path + compliance
    """
    breakFlag = False
    save_past_dict = {}
    log_item = {}
    dmp.step(robot.dt)
    # temporal coupling step
    tau_dmp = dmp.tau + tc.update(dmp, robot.dt) * robot.dt
    dmp.set_tau(tau_dmp)
    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    Z = np.diag(np.array([0.0, 0.0, 1.0, 0.5, 0.5, 0.0]))

    wrench = robot.getWrench()
    save_past_dict["wrench"] = wrench.copy()
    # rolling average
    # wrench = np.average(np.array(past_data['wrench']), axis=0)

    # first-order low pass filtering instead
    # beta is a smoothing coefficient, smaller values smooth more, has to be in [0,1]
    # wrench = args.beta * wrench + (1 - args.beta) * past_data['wrench'][-1]
    wrench = args.beta * wrench + (1 - args.beta) * np.average(
        np.array(past_data["wrench"]), axis=0
    )

    wrench = Z @ wrench
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    dq = robot.getQd()[:6].reshape((6, 1))
    # get joint
    tau = J.T @ wrench
    tau = tau[:6].reshape((6, 1))
    # compute control law:
    # - feedforward the velocity and the force reading
    # - feedback the position
    # TODO solve this q[:6] bs (clean it up)
    vel_cmd = dmp.vel + args.kp * (dmp.pos - q[:6].reshape((6, 1))) + args.alpha * tau
    robot.sendQd(vel_cmd)

    # tau0 is the minimum time needed for dmp
    # 500 is the frequency
    # so we need tau0 * 500 iterations minimum
    if (np.linalg.norm(dmp.vel) < 0.01) and (i > int(args.tau0 * 500)):
        breakFlag = True
    # immediatelly stop if something weird happened (some non-convergence)
    if np.isnan(vel_cmd[0]):
        print("GO NAN FROM INTO VEL_CMD!!! EXITING!!")
        breakFlag = True

    # log what you said you'd log
    # TODO fix the q6 situation (hide this)
    log_item["qs"] = q[:6].reshape((6,))
    log_item["dmp_qs"] = dmp.pos.reshape((6,))
    log_item["dqs"] = dq.reshape((6,))
    log_item["dmp_dqs"] = dmp.vel.reshape((6,))
    log_item["wrench"] = wrench.reshape((6,))
    log_item["tau"] = tau.reshape((6,))

    return breakFlag, save_past_dict, log_item


def write(args, robot: RobotManager, joint_trajectory):
    # create DMP based on the trajectory
    dmp = DMP(joint_trajectory, a_s=1.0)
    if not args.temporal_coupling:
        tc = NoTC()
    else:
        v_max_ndarray = np.ones(robot.n_arm_joints) * robot.max_qd
        a_max_ndarray = np.ones(robot.n_arm_joints) * args.acceleration
        tc = TCVelAccConstrained(
            args.gamma_nominal, args.gamma_a, v_max_ndarray, a_max_ndarray, args.eps_tc
        )

    print("going to starting write position")
    dmp.step(1 / 500)
    first_q = dmp.pos.reshape((6,))
    first_q = list(first_q)
    first_q.append(0.0)
    first_q.append(0.0)
    first_q = np.array(first_q)
    # move to initial pose
    mtool = robot.getT_w_e(q_given=first_q)
    # start a bit above
    go_away_from_plane_transf = pin.SE3.Identity()
    go_away_from_plane_transf.translation[2] = -1 * args.mm_into_board
    mtool = mtool.act(go_away_from_plane_transf)
    if not args.board_wiping:
        compliantMoveL(args, robot, mtool)
    else:
        moveL(args, robot, mtool)

    save_past_dict = {
        "wrench": np.zeros(6),
    }
    # here you give it it's initial value
    log_item = {
        "qs": np.zeros(robot.n_arm_joints),
        "dmp_qs": np.zeros(robot.n_arm_joints),
        "dqs": np.zeros(robot.n_arm_joints),
        "dmp_dqs": np.zeros(robot.n_arm_joints),
        "wrench": np.zeros(6),
        "tau": np.zeros(robot.n_arm_joints),
    }
    # moveJ(args, robot, dmp.pos.reshape((6,)))
    controlLoop = partial(controlLoopWriting, args, robot, dmp, tc)
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    # and now we can actually run
    loop_manager.run()

    print("move a bit back")
    T_w_e = robot.getT_w_e()
    go_away_from_plane_transf = pin.SE3.Identity()
    go_away_from_plane_transf.translation[2] = -0.1
    goal = T_w_e.act(go_away_from_plane_transf)
    compliantMoveL(args, robot, goal)


if __name__ == "__main__":

    #######################################################################
    #                           software setup                            #
    #######################################################################
    args = getArgs()
    if not args.board_wiping:
        assert args.mm_into_board > 0.0 and args.mm_into_board < 5.0
    args.mm_into_board = args.mm_into_board / 1000
    print(args)
    robot = RobotManager(args)
    if args.pinocchio_only:
        rand_pertb = np.zeros(8)
        rand_pertb[:6] = np.random.random(6) * 0.1
        robot.q = (
            np.array([1.32, -1.40, -1.27, -1.157, 1.76, -0.238, 0.0, 0.0]) + rand_pertb
        )
        robot._step()

    #######################################################################
    #          drawing a path, making a joint trajectory for it           #
    #######################################################################

    # draw the path on the screen
    if args.draw_new:
        # pure evil way to solve a bug that was pure evil
        matplotlib.use("tkagg")
        pixel_path = drawPath(args)
        matplotlib.use("qtagg")
    else:
        if not args.board_wiping:
            pixel_path_file_path = "./path_in_pixels.csv"
            pixel_path = np.genfromtxt(pixel_path_file_path, delimiter=",")
        else:
            pixel_path_file_path = "./wiping_path.csv_save"
            pixel_path = np.genfromtxt(pixel_path_file_path, delimiter=",")
    # do calibration if specified
    if args.calibration:
        plane_pose, q_init = calibratePlane(
            args, robot, args.board_width, args.board_height, args.n_calibration_tests
        )
        print("finished calibration")
    else:
        print("using existing plane calibration")
        file = open("./plane_pose.pickle_save", "rb")
        plane_calib_dict = pickle.load(file)
        file.close()
        plane_pose = plane_calib_dict["plane_top_left_pose"]
        q_init = plane_calib_dict["q_init"]
        # stupid fix dw about this it's supposed to be 0.0 on the gripper
        # because we don't use that actually
        q_init[-1] = 0.0
        q_init[-2] = 0.0

    # make the path 3D
    path_points_3D = map2DPathTo3DPlane(pixel_path, args.board_width, args.board_height)
    if args.pick_up_marker:
        # raise NotImplementedError("sorry")
        getMarker(args, robot, None)

    # marker moves between runs, i change markers etc,
    # this goes to start, goes down to touch the board - that's the marker
    # length aka offset (also handles incorrect frames)
    if args.find_marker_offset:
        # find the marker offset
        marker_offset = findMarkerOffset(args, robot, plane_pose)
        robot.sendQd(np.zeros(6))
        print("marker_offset", marker_offset)
        # we're going in a bit deeper
        path_points_3D = path_points_3D + np.array(
            [0.0, 0.0, -1 * marker_offset + args.mm_into_board]
        )
    else:
        print("i hope you know the magic number of marker length + going into board")
        # path = path + np.array([0.0, 0.0, -0.1503])
        marker_offset = 0.066
        path_points_3D = path_points_3D + np.array(
            [0.0, 0.0, -1 * marker_offset + args.mm_into_board]
        )

    # create a joint space trajectory based on the 3D path
    if args.draw_new or args.calibration or args.find_marker_offset:
        path = []
        for i in range(len(path_points_3D)):
            path_pose = pin.SE3.Identity()
            path_pose.translation = path_points_3D[i]
            path.append(plane_pose.act(path_pose))

        if args.viz_test_path:
            print(
                """
        look at the viz now! we're constructing a trajectory for the drawing. 
        it has to look reasonable, otherwise we can't run it!
        """
            )
        clikController = getClikController(args, robot)
        joint_trajectory = clikCartesianPathIntoJointPath(
            args, robot, path, clikController, q_init, plane_pose
        )
        if args.viz_test_path:
            answer = input("did the movement of the manipulator look reasonable? [Y/n]")
            if not (answer == "Y" or answer == "y"):
                print("well if it doesn't look reasonable i'll just exit!")
                answer = False
            else:
                answer = True
        else:
            answer = True
    else:
        joint_trajectory_file_path = "./joint_trajectory.csv"
        joint_trajectory = np.genfromtxt(joint_trajectory_file_path, delimiter=",")

    if answer:
        write(args, robot, joint_trajectory)

    if not args.pinocchio_only:
        robot.stopRobot()

    if args.save_log:
        robot.log_manager.saveLog()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

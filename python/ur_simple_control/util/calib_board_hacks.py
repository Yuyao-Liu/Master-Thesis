import pinocchio as pin
import numpy as np
import time
import copy
from ur_simple_control.managers import RobotManager
from ur_simple_control.clik.clik import moveL, moveUntilContact
from ur_simple_control.basics.basics import freedriveUntilKeyboard
# used to deal with freedrive's infinite while loop
import threading
import argparse
import pickle

"""
general
-----------
Estimate a plane by making multiple contacts with it. 
You need to start with a top left corner of it,
and you thus don't need to find an offset (you have to know it in advance).
TODO: test and make sure the above statement is in fact correct.
Thus the offset does not matter, we only need the angle,
i.e. the normal vector to the plane.
Returns R because that's what's needed to construct the hom transf. mat.
"""

def getBoardCalibrationArgs(parser):
    parser.add_argument('--board-width', type=float, \
            help="width of the board (in meters) the robot will write on", \
            default=0.30)
    parser.add_argument('--board-height', type=float, \
            help="height of the board (in meters) the robot will write on", \
            default=0.30)
    parser.add_argument('--calibration', action=argparse.BooleanOptionalAction, \
            help="whether you want to do calibration", default=False)
    parser.add_argument('--n-calibration-tests', type=int, \
            help="number of calibration tests you want to run", default=10)
    return parser


def fitNormalVector(positions):
    """
    fitNormalVector
    ----------------
    classic least squares fit.
    there's also weighting to make new measurements more important,
    beucase we change the orientation of the end-effector as we go. 
    the change in orientation is done so that the end-effector hits the 
    board at the angle of the board, and thus have consistent measurements.
    """
    positions = np.array(positions)
    # non-weighted least squares as fallback (numerical properties i guess)
    n_non_weighted = np.linalg.lstsq(positions, np.ones(len(positions)), rcond=None)[0]
    n_non_weighted = n_non_weighted / np.linalg.norm(n_non_weighted)
    print("n_non_weighted", n_non_weighted)
    for p in positions:
        print("cdot none", p @ n_non_weighted)
    try:
        # strong
        W = np.diag(np.arange(1, len(positions) + 1))
        n_linearly_weighted = np.linalg.inv(positions.T @ W @ positions) @ positions.T @ W @ np.ones(len(positions))
        n_linearly_weighted = n_linearly_weighted / np.linalg.norm(n_linearly_weighted)
        print("n_linearly_weighed", n_linearly_weighted)
        print("if the following give you roughly the same number, it worked")
        for p in positions:
            print("cdot strong", p @ n_linearly_weighted)
        return n_linearly_weighted
    except np.linalg.LinAlgError:
        print("n_linearly_weighted is singular bruh")
    return n_non_weighted

def constructFrameFromNormalVector(R_initial_estimate, n):
    """
    constructFrameFromNormalVector
    ----------------------------------
    constuct a frame around the found normal vector
    we just assume the x axis is parallel with the robot's x axis
    this is of course completly arbitrary, so
    TODO fix the fact that you just assume the x axis
    or write down why you don't need it (i'm honestly not sure atm, but it is late)
    """
    z_new = n
    x_new = np.array([1.0, 0.0, 0.0])
    y_new = np.cross(x_new, z_new)
    # reshaping so that hstack works as expected
    R = np.hstack((x_new.reshape((3,1)), y_new.reshape((3,1))))
    R = np.hstack((R, z_new.reshape((3,1))))
    # now ensure all the signs are the signs that you want,
    # which we get from the initial estimate (which can not be that off)
    # NOTE this is potentially just an artifact of the previous solution which relied
    # on UR TCP readings which used rpy angles. but it ain't hurting nobody
    # so i'm leaving it.
    R = np.abs(R) * np.sign(R_initial_estimate)
    print('rot mat to new frame:')
    print(*R, sep=',\n')
    return R

def handleUserToHandleTCPPose(args, robot):
    """
    handleUserToHandleTCPPose
    -----------------------------
    1. tell the user what to do with prints, namely where to put the end-effector
      to both not break things and also actually succeed
    2. start freedrive
    3. use some keyboard input [Y/n] as a blocking call,
    4. release the freedrive and then start doing the calibration process
    5. MAKE SURE THE END-EFFECTOR FRAME IS ALIGNED  BY LOOKING AT THE MANIPULATOR VISUALIZER
    """
    print("""
    Whatever code you ran wants you to calibrate the plane on which you will be doing
    your things. Put the end-effector at the top left corner SOMEWHAT ABOVE of the plane 
    where you'll be doing said things. \n
    MAKE SURE THE END-EFFECTOR FRAME IS ALIGNED BY LOOKING AT THE MANIPULATOR VISUALIZER.
    This means x-axis is pointing to the right, and y-axis is pointing down.
    The movement will be defined based on the end-effector's frame so this is crucial.
    Also, make sure the orientation is reasonably correct as that will be 
    used as the initial estimate of the orientation, 
    which is what you will get as an output from this calibration procedure.
    The end-effector will go down (it's TCP z pozitive direction) and touch the plane
    the number of times you specified (if you are not aware of this, check the
    arguments of the program you ran.\n 
    The robot will now enter freedrive mode so that you can manually put
    the end-effector where it's supposed to be.\n 
    When you did it, press 'Y', or press 'n' to exit.
    """)
    while True:
        answer = input("Ready to calibrate or no (no means exit program)? [Y/n]")
        if answer == 'n' or answer == 'N':
            print("""
    The whole program will exit. Change the argument to --no-calibrate or 
    change code that lead you here.
            """)
            exit()
        elif answer == 'y' or answer == 'Y':
            print("""
    The robot will now enter freedrive mode. Put the end-effector to the 
    top left corner of your plane and mind the orientation.
                    """)
            break
        else:
            print("Whatever you typed in is neither 'Y' nor 'n'. Give it to me straight cheif!")
            continue
    print("""
    Entering freedrive. 
    Put the end-effector to the top left corner of your plane and mind the orientation.
    Press Enter to stop freedrive.
    """)
    time.sleep(2)

    freedriveUntilKeyboard(args, robot) 

    while True:
        answer = input("""
    I am assuming you got the end-effector in the correct pose. \n
    Are you ready to start calibrating or not (no means exit)? [Y/n]
    """)
        if answer == 'n' or answer == 'N':
            print("The whole program will exit. Goodbye!")
            exit()
        elif answer == 'y' or answer == 'Y':
            print("Calibration about to start. Have your hand on the big red stop button!")
            time.sleep(2)
            break
        else:
            print("Whatever you typed in is neither 'Y' nor 'n'. Give it to me straight cheif!")
            continue

def calibratePlane(args, robot : RobotManager, plane_width, plane_height, n_tests):
    """
    calibratePlane
    --------------
    makes the user select the top-left corner of the plane in freedrive.
    then we go in the gripper's frame z direction toward the plane.
    we sam
    """
    handleUserToHandleTCPPose(args, robot)
    if args.pinocchio_only:
        robot._step()
    q_init = robot.getQ()
    Mtool = robot.getT_w_e()

    init_pose = copy.deepcopy(Mtool)
    new_pose = copy.deepcopy(init_pose)

    R_initial_estimate = Mtool.rotation.copy()
    print("initial pose estimate:", Mtool)
    R = R_initial_estimate.copy()

    go_away_from_plane_transf = pin.SE3.Identity()
    go_away_from_plane_transf.translation[2] = -0.1
    # used to define going to above new sample point on the board
    go_above_new_sample_transf = pin.SE3.Identity()

    # go in the end-effector's frame z direction 
    # our goal is to align that with board z
    speed = np.zeros(6)
    speed[2] = 0.02

    positions = []
    for i in range(n_tests):
        print("========================================")
        time.sleep(0.01)
        print("iteration number:", i)
        #robot.rtde_control.moveUntilContact(speed)
        moveUntilContact(args, robot, speed)
        # no step because this isn't wrapped by controlLoopManager 
        robot._step()
        q = robot.getQ()
        T_w_e = robot.getT_w_e()
        print("pin:", *T_w_e.translation.round(4), \
                *pin.rpy.matrixToRpy(T_w_e.rotation).round(4))
#        print("ur5:", *np.array(robot.rtde_receive.getActualTCPPose()).round(4))

        positions.append(copy.deepcopy(T_w_e.translation))
        if i < n_tests -1:
            current_pose = robot.getT_w_e()
            # go back up 
            new_pose = current_pose.act(go_away_from_plane_transf)
            moveL(args, robot, new_pose)

            # MAKE SURE THE END-EFFECTOR FRAME IS ALIGNED AS INSTRUCTED:
            # positive x goes right, positive y goes down
            # and you started in the top left corner
            print("going to new pose for detection", new_pose)
            new_pose = init_pose.copy()
            go_above_new_sample_transf.translation[0] = np.random.random() * plane_width
            go_above_new_sample_transf.translation[1] = np.random.random() * plane_height
            new_pose = new_pose.act(go_above_new_sample_transf)
            moveL(args, robot, new_pose)
            # fix orientation
            new_pose.rotation = R
            print("updating orientation")
            moveL(args, robot, new_pose)
        # skip the first one
        if i > 2:
            n = fitNormalVector(positions)
            R = constructFrameFromNormalVector(R_initial_estimate, n)
            speed = np.zeros(6)
            speed[2] = 0.02

    print("finished estimating R")

    current_pose = robot.getT_w_e()
    new_pose = current_pose.copy()
    # go back up
    new_pose = new_pose.act(go_away_from_plane_transf)
    moveL(args, robot, new_pose)
    # go back to the same spot
    new_pose.translation[0] = init_pose.translation[0]
    new_pose.translation[1] = init_pose.translation[1]
    new_pose.translation[2] = init_pose.translation[2]
    # but in new orientation
    new_pose.rotation = R
    print("going back to initial position with fitted R")
    moveL(args, robot, new_pose)
    
    print("i'll estimate the translation vector to board beginning now \
           that we know we're going straight down")
    speed = np.zeros(6)
    speed[2] = 0.02

    moveUntilContact(args, robot, speed)

    q = robot.getQ()
    pin.forwardKinematics(robot.model, robot.data, np.array(q))
    Mtool = robot.getT_w_e(q_given=q)
    translation = Mtool.translation.copy()
    print("got translation vector, it's:", translation)


    moveL(args, robot, new_pose)
    q = robot.getQ()
    init_q = copy.deepcopy(q)
    print("went back up, saved this q as initial q")
    
    # put the speed slider back to its previous value
#    robot.setSpeedSlider(old_speed_slider)
    print('also, the translation vector is:', translation)
    if not args.pinocchio_only:
        file_path = './plane_pose.pickle'
    else:
        file_path = './plane_pose_sim.pickle'
    log_file = open(file_path, 'wb')
    plane_pose = pin.SE3(R, translation)
    log_item = {'plane_top_left_pose': plane_pose, 'q_init': q_init.copy()}
    pickle.dump(log_item, log_file)
    log_file.close()
    return plane_pose, q_init

# TODO: update for the current year
#if __name__ == "__main__":
#    robot = RobotManager()
#    # TODO make this an argument
#    n_tests = 10
#    # TODO: 
#    # - tell the user what to do with prints, namely where to put the end-effector
#    #   to both not break things and also actually succeed
#    # - start freedrive
#    # - use some keyboard input [Y/n] as a blocking call,
#    #   release the freedrive and then start doing the calibration process
#    calibratePlane(robot, n_tests)

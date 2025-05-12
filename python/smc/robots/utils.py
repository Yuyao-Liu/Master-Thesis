from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.robots.implementations import *

import numpy as np
import pinocchio as pin
import argparse


def getMinimalArgParser():
    """
    getDefaultEssentialArgs
    ------------------------
    returns a parser containing:
        - essential arguments (run in real or in sim)
        - parameters for (compliant)moveJ
        - parameters for (compliant)moveL
    """
    parser = argparse.ArgumentParser(
        description="Run something with \
            Simple Manipulator Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ########################################################
    #  general arguments: robot, simulation, plotting etc  #
    ########################################################
    parser.add_argument(
        "--robot",
        type=str,
        help="which robot you're running or simulating",
        default="ur5e",
        choices=["ur5e", "heron", "yumi", "myumi", "mir"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="control mode the robot should run in, i.e. select if you're controlling only a subset of the available joints. most of these only make sense for mobile robots with arms of course (you get an error upon an erroneous selection)",
        default="whole_body",
        choices=[
            "whole_body",
            "base_only",
            "upper_body",
            "left_arm_only",
            "right_arm_only",
        ],
    )
    parser.add_argument(
        "--real",
        action=argparse.BooleanOptionalAction,
        help="whether you're running on the real robot or not",
        default=False,
    )
    # if this ends up working, replace --real with --mode, which can be {real, integration (simulation), debugging}
    parser.add_argument(
        "--robot-ip",
        type=str,
        help="robot's ip address (only needed if running on the real robot)",
        default="192.168.1.102",
    )
    parser.add_argument(
        "--ctrl-freq",
        type=int,
        help="frequency of the control loop. select -1 if you want to go as fast as possible (useful for running tests in sim)",
        default=500,
    )
    parser.add_argument(
        "--visualizer",
        action=argparse.BooleanOptionalAction,
        help="whether you want to visualize the manipulator and workspace with meshcat",
        default=True,
    )
    parser.add_argument(
        "--viz-update-rate",
        type=int,
        help="frequency of visual updates. visualizer and plotter update every viz-update-rate^th iteration of the control loop. put to -1 to get a reasonable heuristic",
        default=-1,
    )
    parser.add_argument(
        "--plotter",
        action=argparse.BooleanOptionalAction,
        help="whether you want to have some real-time matplotlib graphs (parts of log_dict you select)",
        default=True,
    )
    parser.add_argument(
        "--gripper",
        type=str,
        help="gripper you're using (no gripper is the default)",
        default="none",
        choices=["none", "robotiq", "onrobot", "rs485"],
    )
    # TODO: make controlloop manager run in a while True loop and remove this
    # ==> max-iterations actually needs to be an option. sometimes you want to simulate and exit
    #     if the convergence does not happen in a reasonable amount of time,
    #     ex. goal outside of workspace has been passed or something
    # =======> if it's set to 0 then the loops run infinitely long
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="maximum allowable iteration number (it runs at 500Hz)",
        default=100000,
    )
    parser.add_argument(
        "--start-from-current-pose",
        action=argparse.BooleanOptionalAction,
        help="if connected to the robot, read the current pose and set it as the initial pose for the robot. \
                 very useful and convenient when running simulation before running on real",
        default=False,
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        help="robot's joints acceleration. scalar positive constant, max 1.7, and default 0.3. \
                   BE CAREFUL WITH THIS. the urscript doc says this is 'lead axis acceleration'.\
                   TODO: check what this means",
        default=0.3,
    )
    parser.add_argument(
        "--max-v-percentage",
        type=float,
        help="select the percentage of the maximum joint velocity the robot can achieve to be the control input maximum (control inputs are clipped to perc * max_v)",
        default=0.3,
    )
    parser.add_argument(
        "--debug-prints",
        action=argparse.BooleanOptionalAction,
        help="print some debug info",
        default=False,
    )
    parser.add_argument(
        "--save-log",
        action=argparse.BooleanOptionalAction,
        help="whether you want to save the log of the run. it saves \
                        what you pass to ControlLoopManager. check other parameters for saving directory and log name.",
        default=False,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="path to where you store your logs. default is ./data, but if that directory doesn't exist, then /tmp/data is created and used.",
        default="./data",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="name the whole run/experiment (name of log file). note that indexing of runs is automatic and under a different argument.",
        default="latest_run",
    )
    parser.add_argument(
        "--index-runs",
        action=argparse.BooleanOptionalAction,
        help="if you want more runs of the same name, this option will automatically assign an index to a new run (useful for data collection).",
        default=False,
    )
    parser.add_argument(
        "--past-window-size",
        type=int,
        help="how many timesteps of past data you want to save",
        default=5,
    )
    # maybe you want to scale the control signal (TODO: NOT HERE)
    parser.add_argument(
        "--controller-speed-scaling",
        type=float,
        default="1.0",
        help="not actually_used atm",
    )
    ########################################
    #  environment interaction parameters  #
    ########################################
    parser.add_argument(
        "--contact-detecting-force",
        type=float,  # default=1.3, help='the force used to detect contact (collision) in the moveUntilContact function')
        default=2.8,
        help="the force used to detect contact (collision) in the moveUntilContact function",
    )
    parser.add_argument(
        "--minimum-detectable-force-norm",
        type=float,
        help="we need to disregard noise to converge despite filtering. \
                  a quick fix is to zero all forces of norm below this argument threshold.",
        default=3.0,
    )
    # TODO make this work without parsing (or make it possible to parse two times)
    # if (args.gripper != "none") and args.simulation:
    #    raise NotImplementedError('Did not figure out how to put the gripper in \
    #            the simulation yet, sorry :/ . You can have only 1 these flags right now')
    parser.add_argument(
        "--visualize-collision-approximation",
        action=argparse.BooleanOptionalAction,
        help="whether you want to visualize the collision approximation used in controllers with obstacle avoidance",
        default=False,
    )
    return parser


# TODO: make robot-independent
def defineGoalPointCLI(robot):
    """
    defineGoalPointCLI
    ------------------
    get a nice TUI-type prompt to put in a frame goal for p2p motion.
    --> best way to handle the goal is to tell the user where the gripper is
        in both UR tcp frame and with pinocchio and have them
        manually input it when running.
        this way you force the thinking before the moving,
        but you also get to view and analyze the information first
    TODO get the visual thing you did in ivc project with sliders also.
    it's just text input for now because it's totally usable, just not superb.
    but also you do want to have both options. obviously you go for the sliders
    in the case you're visualizing, makes no sense otherwise.
    """
    robot._step()
    robot._step()
    # define goal
    T_w_goal = robot.T_w_e
    print("You can only specify the translation right now.")
    if robot.args.real:
        print(
            "In the following, first 3 numbers are x,y,z position, and second 3 are r,p,y angles"
        )
        print(
            "Here's where the robot is currently. Ensure you know what the base frame is first."
        )
        print(
            "base frame end-effector pose from pinocchio:\n",
            *robot.data.oMi[6].translation.round(4),
            *pin.rpy.matrixToRpy(robot.data.oMi[6].rotation).round(4),
        )
        print("UR5e TCP:", *np.array(robot._rtde_receive.getActualTCPPose()).round(4))
    # remain with the current orientation
    # TODO: add something, probably rpy for orientation because it's the least number
    # of numbers you need to type in
    # this is a reasonable way to do it too, maybe implement it later
    # T_w_goal.translation = T_w_goal.translation + np.array([0.0, 0.0, -0.1])
    # do a while loop until this is parsed correctly
    while True:
        goal = input(
            "Please enter the target end-effector position in the x.x,y.y,z.z format: "
        )
        try:
            e = "ok"
            goal_list = goal.split(",")
            for i in range(len(goal_list)):
                goal_list[i] = float(goal_list[i])
        except:
            e = exc_info()
            print("The input is not in the expected format. Try again.")
            print(e)
        if e == "ok":
            T_w_goal.translation = np.array(goal_list)
            break
    print("this is goal pose you defined:\n", T_w_goal)

    if robot.args.visualizer:
        robot.visualizer_manager.sendCommand({"Mgoal": T_w_goal})
    return T_w_goal


# TODO: finish
def getRobotFromArgs(args: argparse.Namespace) -> AbstractRobotManager:
    if args.robot == "ur5e":
        if args.real:
            return RealUR5eRobotManager(args)
        else:
            return SimulatedUR5eRobotManager(args)
    if args.robot == "heron":
        if args.real:
            pass
            # TODO: finish it
            # return RealHeronRobotManager(args)
        else:
            return SimulatedHeronRobotManager(args)
    if args.robot == "yumi":
        if args.real:
            pass
            # TODO: finish it
            # return RealYuMiRobotManager(args)
        else:
            return SimulatedYuMiRobotManager(args)
    if args.robot == "myumi":
        if args.real:
            return RealMobileYuMiRobotManager(args)
        else:
            return SimulatedMobileYuMiRobotManager(args)
    if args.robot == "mir":
        if args.real:
            pass
            # TODO: finish it
            # return RealMirRobotManager(args)
        else:
            return SimulatedMirRobotManager(args)
    raise NotImplementedError(
        f"robot {args.robot} is not supported! run the script you ran with --help to see what's available"
    )


#    if args.robot == "mir":
#        return RealUR5eRobotManager(args)
#    if args.robot == "yumi":
#        return RealUR5eRobotManager(args)

from smc import getMinimalArgParser, getRobotFromArgs
from smc.util.define_random_goal import getRandomlyGeneratedGoal
from smc.control.cartesian_space import getClikArgs
from smc.robots.utils import defineGoalPointCLI
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL

import argparse
import numpy as np


def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.description = "Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!"
    parser = getClikArgs(parser)
    parser.add_argument(
        "--randomly-generate-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if true, the target pose is randomly generated, if false you type it target translation in via text input",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    robot = getRobotFromArgs(args)
    if (not args.real) and args.randomly_generate_goal:
        T_w_goal = getRandomlyGeneratedGoal(args)
        if args.visualizer:
            robot.visualizer_manager.sendCommand({"Mgoal": T_w_goal})
    else:
        if args.real and args.randomly_generate_goal:
            print("Ain't no way you're going to a random goal on the real robot!")
            print("Look at the current pose, define something appropriate manually")
        T_w_goal = defineGoalPointCLI(robot)
        T_w_goal.rotation = np.eye(3)
    # compliantMoveL(args, robot, Mgoal)
    print(robot.mode)
    robot.closeGripper()
    robot.openGripper()
    # moveL(args, robot, T_w_goal)

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()


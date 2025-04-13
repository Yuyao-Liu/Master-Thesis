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
    args.robot = 'heron'
    # robot='ur5e', mode='whole_body', real=False, robot_ip='192.168.1.102', ctrl_freq=500, visualizer=True, viz_update_rate=-1, plotter=True, gripper='none', max_iterations=100000, start_from_current_pose=False, acceleration=0.3, max_v_percentage=0.3, debug_prints=False, save_log=False, save_dir='./data', run_name='latest_run', index_runs=False, past_window_size=5, controller_speed_scaling=1.0, contact_detecting_force=2.8, minimum_detectable_force_norm=3.0, visualize_collision_approximation=False, goal_error=0.01, tikhonov_damp=0.001, ik_solver='dampedPseudoinverse', alpha=0.01, beta=0.01, kp=1.0, kv=0.001, z_only=False, max_init_clik_iterations=10000, max_running_clik_iterations=1000, viz_test_path=False, randomly_generate_goal=False
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
    moveL(args, robot, T_w_goal)
    robot.closeGripper()
    robot.openGripper()

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()

# PYTHON_ARGCOMPLETE_OK
import numpy as np
import pinocchio as pin
import argcomplete, argparse
from managers import getMinimalArgParser, RobotManager
from clik import (
    getClikArgs,
    getClikController,
    controlLoopClik,
    moveL,
)
from ur_simple_control.util.define_random_goal import getRandomlyGeneratedGoal


def get_args():
    parser = getMinimalArgParser()
    parser.description = "Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!"
    parser = getClikArgs(parser)
    parser.add_argument(
        "--randomly-generate-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if true, rand generate a goal, if false you type it in via text input",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    args.robot = 'heron'
    # (robot='heron', simulation=False, robot_ip='192.168.1.102', pinocchio_only=True, ctrl_freq=500, fast_simulation=False, visualizer=True, plotter=True, gripper='none', max_iterations=100000, speed_slider=1.0, start_from_current_pose=False, acceleration=0.3, max_qd=0.5, debug_prints=False, save_log=False, save_dir='./data', run_name='latest_run', index_runs=False, past_window_size=5, controller_speed_scaling=1.0, contact_detecting_force=2.8, minimum_detectable_force_norm=3.0, visualize_collision_approximation=True, goal_error=0.01, tikhonov_damp=0.001, clik_controller='dampedPseudoinverse', alpha=0.01, beta=0.01, max_init_clik_iterations=10000, max_running_clik_iterations=1000, viz_test_path=False, randomly_generate_goal=False
    args.simulation = True

    robot = RobotManager(args)
    if args.randomly_generate_goal:
        Mgoal = getRandomlyGeneratedGoal(args)
        if args.visualizer:
            robot.visualizer_manager.sendCommand({"Mgoal": Mgoal})
    else:
        Mgoal = robot.defineGoalPointCLI()
        Mgoal.rotation = np.eye(3)
    # compliantMoveL(args, robot, Mgoal)
    moveL(args, robot, Mgoal)

    robot.closeGripper()
    robot.openGripper()
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.saveLog()
    # loop_manager.stopHandler(None, None)

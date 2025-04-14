from smc import getMinimalArgParser, getRobotFromArgs
from smc.util.define_random_goal import getRandomlyGeneratedGoal
from smc.control.cartesian_space import getClikArgs
from smc.robots.utils import defineGoalPointCLI
from smc.control.cartesian_space.cartesian_space_point_to_point import (
    moveL_only_arm,
    park_base,
    move_u_ref
    )
import argparse
import numpy as np
import pinocchio as pin
import time

class Adaptive_controller_manager:
    def __init__(self, robot, alpha=1, beta=1, gamma=1):
        self.robot = robot
        # hyper-parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gain_matrix = np.eye(3)
        # target
        self.f_d = np.zeros(3)
        self.v_d = 0.1
        # initalize parameter
        self.err_sum = np.zeros(3)
        self.x_h_oe = np.array([1, 0, 0])
        self.v_f = np.zeros(3)
        self.v_ref = np.zeros(3)
        
    @staticmethod
    def Proj(x):
        return np.eye(3) - np.outer(x, x)
    
    def get_v_ref(self):
        self.get_v_f()
        self.v_ref = self.v_d * self.x_h_oe - self.Proj(self.x_h_oe) @ self.v_f
        self.get_x_h_oe()
        return self.v_ref
    
    def get_v_f(self):
        # TODO f needs to be in the ee frame
        f = self.robot.getWrench()
        f = f[:3]
        # print(f)
        f_error = f - self.f_d
        self.err_sum += self.Proj(self.x_h_oe) @ f_error 
        self.v_f = self.alpha * f_error + self.beta * self.err_sum
    
    def get_x_h_oe(self):
        v_ref_s = np.sign(np.dot(self.x_h_oe.T, self.v_ref)) * np.linalg.norm(self.v_ref)
        k_oe = np.cross(self.gain_matrix @ self.x_h_oe, self.v_ref)
        self.x_h_oe = -self.gamma * v_ref_s * self.Proj(self.x_h_oe) @ self.v_f - v_ref_s * np.cross(self.x_h_oe, k_oe)
        self.x_h_oe = self.x_h_oe / np.linalg.norm(self.x_h_oe)

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
    args.robot = "heron"
    # args.robot = "ur5e"
    robot = getRobotFromArgs(args)
    # args.clik_controller = "dampedPseudoinverse"
    # args.clik_controller = "dPi_adptive_manipulability"
    # args.clik_controller = "dPi_Weighted"
    # args.clik_controller = "taskPriorityInverse"
    # args.clik_controller = "dPi_Weighted_nullspace"
    args.ik_solver = "keep_distance_nullspace"
    
    args.real=False
    args.visualizer=True
    args.plotter = True
    args.max_v_percentage=1
    # robot='ur5e', mode='whole_body', real=False, robot_ip='192.168.1.102', ctrl_freq=500, visualizer=True, viz_update_rate=-1, plotter=True, gripper='none', max_iterations=100000, start_from_current_pose=False, acceleration=0.3, max_v_percentage=0.3, debug_prints=False, save_log=False, save_dir='./data', run_name='latest_run', index_runs=False, past_window_size=5, controller_speed_scaling=1.0, contact_detecting_force=2.8, minimum_detectable_force_norm=3.0, visualize_collision_approximation=False, goal_error=0.01, tikhonov_damp=0.001, ik_solver='dampedPseudoinverse', alpha=0.01, beta=0.01, kp=1.0, kv=0.001, z_only=False, max_init_clik_iterations=10000, max_running_clik_iterations=1000, viz_test_path=False, randomly_generate_goal=False
    Adaptive_controller = Adaptive_controller_manager(robot)
    # move to a proper position for initialization
    translation = np.array([-2.0,-2.5,0.5])
    # translation = np.array([-0.0,-0.5,0.5])
    theta = np.radians(90)
    # rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [-np.sin(theta), -np.cos(theta), 0], [0, 0, -1]])
    rotation = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    Mgoal = pin.SE3(rotation, translation)
    # Mgoal = getRandomlyGeneratedGoal(args)

    if args.visualizer:
        robot.visualizer_manager.sendCommand({"Mgoal": Mgoal})
        
    time.sleep(5)
    park_base(args, robot, (-1.5, -2.5, 0))
    moveL_only_arm(args, robot, Mgoal)
    print('moveL done')
    move_u_ref(args, robot, Adaptive_controller)
    robot.closeGripper()
    robot.openGripper()

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()

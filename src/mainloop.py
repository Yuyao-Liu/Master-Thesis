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
import scipy.io as sio
import os

class Adaptive_controller_manager:
    def __init__(self, robot, alpha=1, beta=1, gamma=10000):
        self.robot = robot
        self.robot.v_ee = 0
        # hyper-parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gain_matrix = np.eye(3)*10000
        # target
        self.f_d = np.zeros(3)
        self.v_d = np.pi/40
        # initalize parameter
        self.err_sum = np.zeros(3)
        self.x_h = np.array([1.3, -0.1, 0.1])
        self.v_f = np.zeros(3)
        self.v_ref = np.array([1, 0, 0])
        self.k_h = np.array([0.1, 1.2, 0.1])
        self.time = time.perf_counter()
        self.starttime = time.perf_counter()
        self.x_h_history = []
        self.k_history = []
        self.v_ref_history = []
        
    @staticmethod
    def Proj(x):
        return np.eye(3) - np.outer(x, x)
    
    def get_v_ref(self):
        self.get_v_f()
        self.v_ref = self.v_d * self.x_h - self.Proj(self.x_h) @ self.v_f
        self.get_x_h()
        return self.v_ref
    
    def get_v_f(self):
        # TODO f needs to be in the ee frame
        f = self.robot.getWrench()
        f = f[:3]
        # print(f)
        f_error = f - self.f_d
        self.err_sum += self.Proj(self.x_h) @ f_error 
        self.v_f = self.alpha * f_error + self.beta * self.err_sum
    
    def update_time(self):
        self.time = time.perf_counter()
        
    def get_x_h(self):
        # save data
        self.x_h_history.append(self.x_h.copy())
        self.k_history.append(self.k_h.copy())
        
        dt = time.perf_counter() - self.time
        self.time = time.perf_counter()
        
        T_w_e = self.robot.T_w_e
        if robot.task == 3:
            self.v_ref = -T_w_e.rotation[:, 1] * abs(self.robot.v_ee)
            self.v_ref_history.append((-T_w_e.rotation[:, 1]).copy())
        else:
            self.v_ref = -T_w_e.rotation[:, 2] * abs(self.robot.v_ee)
            self.v_ref_history.append((-T_w_e.rotation[:, 2]).copy())
            
        vf = self.v_ref   
        v_ref_norm = np.sign(np.dot(self.x_h.T, self.v_ref)) * np.linalg.norm(vf)

        k_dot = self.gain_matrix @ np.cross(self.x_h, vf)
        self.k_h = self.k_h + dt * k_dot

        x_h_dot = self.gamma * v_ref_norm * self.Proj(self.x_h) @ vf - v_ref_norm * np.cross(self.x_h, self.k_h)
        self.x_h = self.x_h + dt * x_h_dot
        self.x_h = self.x_h / np.linalg.norm(self.x_h)

        # print(self.time - self.starttime)
        if len(self.x_h_history) == 2e4:
            self.save_history_to_mat("log.mat")
        return self.x_h

    def save_history_to_mat(self, filename):
        sio.savemat(filename, {
            "x_h_oe_history": np.array(self.x_h_history),
            "k_oe_history": np.array(self.k_history),
            "v_ref_history": np.array(self.v_ref_history)
        })
        print(f"Data are saved in {filename}")
        
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
    args.ik_solver = "keep_distance_nullspace"
    
    args.real=False
    args.visualizer=True
    args.plotter = False
    args.max_v_percentage=5
    # robot='ur5e', mode='whole_body', real=False, robot_ip='192.168.1.102', ctrl_freq=500, visualizer=True, viz_update_rate=-1, plotter=True, gripper='none', max_iterations=100000, start_from_current_pose=False, acceleration=0.3, max_v_percentage=0.3, debug_prints=False, save_log=False, save_dir='./data', run_name='latest_run', index_runs=False, past_window_size=5, controller_speed_scaling=1.0, contact_detecting_force=2.8, minimum_detectable_force_norm=3.0, visualize_collision_approximation=False, goal_error=0.01, tikhonov_damp=0.001, ik_solver='dampedPseudoinverse', alpha=0.01, beta=0.01, kp=1.0, kv=0.001, z_only=False, max_init_clik_iterations=10000, max_running_clik_iterations=1000, viz_test_path=False, randomly_generate_goal=False
    Adaptive_controller = Adaptive_controller_manager(robot)
    # move to a proper position for initialization
    translation = np.array([-2.0,-2.5,1.3])
    # translation = np.array([-0.0,-0.5,0.5])
    theta = np.radians(90)
    # rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [-np.sin(theta), -np.cos(theta), 0], [0, 0, -1]])
    rotation = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    handle_pose = pin.SE3(rotation, translation)
    # Mgoal = getRandomlyGeneratedGoal(args)
    robot.handle_pose = handle_pose
    robot.angle_desired = 120
    if args.visualizer:
        robot.visualizer_manager.sendCommand({"Mgoal": handle_pose})
    robot.task = 1
    # time.sleep(5)
    park_base(args, robot, (-1.2, -2.5, 0), run=True)
    moveL_only_arm(args, robot, handle_pose, run=True)
    print('moveL done')
    Adaptive_controller.update_time()
    move_u_ref(args, robot, Adaptive_controller, run=True)
    robot.closeGripper()
    robot.openGripper()
    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()

from smc import getMinimalArgParser, getRobotFromArgs
from smc.util.define_random_goal import getRandomlyGeneratedGoal
from smc.control.cartesian_space import getClikArgs
from smc.robots.utils import defineGoalPointCLI
from smc.control.cartesian_space.cartesian_space_point_to_point import (
    moveL_only_arm,
    park_base,
    move_u_ref
    )
from smc.robots.abstract_robotmanager import AbstractRobotManager

from smc.multiprocessing.smc_heron_node import get_args, SMCHeronNode, GazeboSMCHeronNode
from smc.robots.implementations.heron import RealHeronRobotManager, GazeboHeronRobotManager

import argparse
import numpy as np
import pinocchio as pin
import time
import scipy.io as sio
import os
import rclpy
from rclpy.executors import MultiThreadedExecutor

class Adaptive_controller_manager:
    def __init__(self, robot, alpha=1, beta=1, gamma=2000):
        self.robot = robot
        self.robot.v_ee = 0
        # hyper-parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gain_matrix = np.eye(3)*2000
        # target
        self.f_d = np.zeros(3)
        self.v_d = np.pi/40
        # initalize parameter
        self.err_sum = np.zeros(3)
        self.x_h = np.array([1, -2, 3])
        self.v_f = np.zeros(3)
        self.v_ref = np.array([1, 0, 0])
        self.k_h = np.array([3, -1, 2])
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
        self.v_ref = -T_w_e.rotation[:, 2] * abs(self.robot.v_ee)
        vf = self.v_ref
        self.v_ref_history.append((-T_w_e.rotation[:, 2]).copy())
        
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


if __name__ == "__main__":
    args = None
    args_smc = get_args()
    args_smc.robot = "heron"
    args_smc.ik_solver = "keep_distance_nullspace"
    real = False
    if real:
        args_smc.real=True
        args_smc.unreal = False
        args_smc.gazebo = False
    else:
        args_smc.real=False
        args_smc.unreal = True
        args_smc.gazebo = True
    args_smc.robot_ip = "192.168.04"
    args_smc.visualizer=True
    args_smc.plotter = False
    args_smc.max_v_percentage = 0.1
    # args_smc.max_v_percentage=5
    # assert args_smc.robot == "heron"
    # robot = RealHeronRobotManager(args_smc)
    if args_smc.gazebo:
        robot = GazeboHeronRobotManager(args_smc)
    else:
        robot = RealHeronRobotManager(args_smc) 
    robot._step()
    modes_and_loops = []
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
    if args_smc.visualizer:
        robot.visualizer_manager.sendCommand({"Mgoal": handle_pose})
    robot.task = 1   
    # time.sleep(5)
    mode_1 = AbstractRobotManager.control_mode.whole_body
    loop_1 = park_base(args_smc, robot, (-1.2, -2.5, 0))
    # modes_and_loops.append((mode_1, loop_1))
    
    mode_2 = AbstractRobotManager.control_mode.whole_body
    loop_2 = moveL_only_arm(args_smc, robot, handle_pose)
    modes_and_loops.append((mode_2, loop_2))
    
    mode_3 = AbstractRobotManager.control_mode.whole_body
    loop_3 = move_u_ref(args_smc, robot, Adaptive_controller)
    # modes_and_loops.append((mode_3, loop_3))
    
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    if args_smc.gazebo:
        node = GazeboSMCHeronNode(args_smc, robot, modes_and_loops)
    else:
        node = SMCHeronNode(args_smc, robot, modes_and_loops)
    executor.add_node(node)
    executor.spin()

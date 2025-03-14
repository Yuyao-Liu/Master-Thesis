# PYTHON_ARGCOMPLETE_OK
import numpy as np
import pinocchio as pin
import argcomplete, argparse
from functools import partial
import time
import copy
import importlib
from managers import getMinimalArgParser, RobotManager, ControlLoopManager
from clik import (
    getClikArgs,
    getClikController,
    controlLoopClik,
    moveL,
    dampedPseudoinverse,
    dPi_Weighted,
    dPi_Weighted_nullspace,
    compute_ee2basedistance,
    parking_base,
    park_base,
    move_u_ref,
)
if importlib.util.find_spec("shapely"):
    from ur_simple_control.path_generation.planner import (
        path2D_to_timed_SE3,
        pathPointFromPathParam,
    )
from ur_simple_control.util.define_random_goal import getRandomlyGeneratedGoal


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
    alpha = 1
    beta = 1
    
    args = get_args()
    args.robot = "heron"
    # args.robot = "ur5e"
    
    # args.clik_controller = "dampedPseudoinverse"
    # args.clik_controller = "dPi_adptive_manipulability"
    # args.clik_controller = "dPi_Weighted"
    # args.clik_controller = "taskPriorityInverse"
    args.clik_controller = "dPi_Weighted_nullspace"
    
    args.pinocchio_only=True
    args.visualizer=True
    args.plotter = False
    # (robot='heron', simulation=False, robot_ip='192.168.1.102', pinocchio_only=True, ctrl_freq=500, fast_simulation=False, visualizer=True, plotter=True, gripper='none', max_iterations=100000, speed_slider=1.0, start_from_current_pose=False, acceleration=0.3, max_qd=0.5, debug_prints=False, save_log=False, save_dir='./data', run_name='latest_run', index_runs=False, past_window_size=5, controller_speed_scaling=1.0, contact_detecting_force=2.8, minimum_detectable_force_norm=3.0, visualize_collision_approximation=True, goal_error=0.01, tikhonov_damp=0.001, clik_controller='dampedPseudoinverse', alpha=0.01, beta=0.01, max_init_clik_iterations=10000, max_running_clik_iterations=1000, viz_test_path=False, randomly_generate_goal=False
    args.simulation = True
    robot = RobotManager(args)
    Adaptive_controller = Adaptive_controller_manager(robot)
    
    # move to a proper position for initialization
    translation = np.array([-2.0,-2.5,0.3])
    theta = np.radians(90)
    # rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [-np.sin(theta), -np.cos(theta), 0], [0, 0, -1]])
    rotation = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    Mgoal = pin.SE3(rotation, translation)
    # Mgoal = getRandomlyGeneratedGoal(args)

    if args.visualizer:
        robot.visualizer_manager.sendCommand({"Mgoal": Mgoal})
    time.sleep(5)
    park_base(args, robot, (-1.2, -2.5, 0))
    moveL(args, robot, Mgoal)
    print('moveL done')
    print(robot.getQ())
    move_u_ref(args, robot, Adaptive_controller)
    robot.closeGripper()
    robot.openGripper()
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot.log_manager.saveLog()
    # loop_manager.stopHandler(None, None)

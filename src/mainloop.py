# PYTHON_ARGCOMPLETE_OK
import numpy as np
import pinocchio as pin
import argcomplete, argparse
from functools import partial
import copy
import importlib
from managers import getMinimalArgParser, RobotManager, ControlLoopManager
from clik import (
    getClikArgs,
    getClikController,
    controlLoopClik,
    moveL,
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

def dampedPseudoinverse(tikhonov_damp, J, u_ref):
    qd = (
        J.T
        @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * tikhonov_damp)
        @ u_ref
    )
    # print(u_ref)
    return qd

# Transform a velocity vector from the world frame to the end-effector (EE) frame
def transform_velocity_to_e(robot: RobotManager):
    """
    Transform a velocity vector from the world frame to the end-effector (EE) frame.

    Parameters:
    - T_w_e: SE3 transformation matrix (Pinocchio SE3 object), representing the pose of the end-effector in the world frame.
    - u_ref_w: 6D velocity vector [v_x, v_y, v_z, ω_x, ω_y, ω_z] in the world frame.

    Returns:
    - u_ref_e: 6D velocity vector in the EE frame.
    """
    # Extract the rotation matrix R_w_e (3x3)
    T_w_e = robot.getT_w_e()
    R_w_e = T_w_e.rotation  

    # Extract linear and angular velocity components in the world frame
    u_ref_w = robot.u_ref_w
    v_w = u_ref_w[:3]  # Linear velocity in the world frame
    omega_w = u_ref_w[3:]  # Angular velocity in the world frame

    # Transform velocities to the EE frame
    v_e = R_w_e.T @ v_w  # Linear velocity in the EE frame
    omega_e = R_w_e.T @ omega_w  # Angular velocity in the EE frame

    # Combine into a 6D velocity vector
    u_ref_e = np.hstack((v_e, omega_e))
    
    return u_ref_e

# controlLoopClik_u_ref for move_u_ref
def controlLoopClik_u_ref(robot: RobotManager, Adaptive_controller: Adaptive_controller_manager, clik_controller, i, past_data):
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.getQ()
    
    # TODO set a proper omega
    v_ref = Adaptive_controller.get_v_ref()
    robot.u_ref_w = np.hstack((v_ref, np.zeros(3)))
    # print(robot.u_ref_w)
    # Convert the twist u_ref_w (6D) in world frame to ee frame
    u_ref_e = transform_velocity_to_e(robot)
    # print(u_ref_e)
    err_vector = u_ref_e
    
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    # compute the joint velocities based on controller you passed
    # qd = clik_controller(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    qd = clik_controller(J, err_vector)
    # print(qd)
    if qd is None:
        print("the controller you chose didn't work, using dampedPseudoinverse instead")
        qd = dampedPseudoinverse(1e-2, J, err_vector)
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    log_item["err_norm"] = np.linalg.norm(err_vector).reshape((1,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    save_past_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    return breakFlag, save_past_item, log_item

# move in a reference twist
def move_u_ref(args, robot: RobotManager, Adaptive_controller: Adaptive_controller_manager):
    """
    move_u_ref
    -----
    come from moveL
    send a reference twist u_ref_w instead.
    """
    clik_controller = getClikController(args, robot)
    controlLoop = partial(controlLoopClik_u_ref, robot, Adaptive_controller, clik_controller)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "dqs_cmd": np.zeros(robot.model.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {
        "dqs_cmd": np.zeros(robot.model.nv),
    }
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()

# def Proj(x):
#     x_proj = np.eye(3) - np.outer(x,x)
#     return x_proj

# # TODO compute v_ref
# def get_v_ref(v_d, v_f, x_h_oe):
#     v_f = get_v_f
#     x_h_oe = get_x_h_oe
#     v_ref = v_d * x_h_oe - Proj(x_h_oe) @ v_f
#     return v_ref

# # TODO compute v_f
# def get_v_f(f_d, robot: RobotManager, x_h_oe):
#     # hyper-parameters
#     f = robot.getWrench()
#     f_error = f - f_d
#     err_sum += Proj(x_h_oe) @ f_error
#     alpha = 1
#     beta = 1
#     v_f = alpha * f_error + beta * err_sum
#     return v_f

# # TODO compute motion direction x_h_oe
# def get_x_h_oe(x_h_oe, v_f, v_ref):
#     # hyper-parameters
#     gamma = 1
#     A = np.eye(3)
    
#     v_ref_s = np.sign(np.dot(x_h_oe, v_ref)) * np.linalg.norm(v_ref)
#     k_oe = np.cross(A @ x_h_oe, v_ref)
#     x_h_oe = -gamma * v_ref_s * Proj(x_h_oe) @ v_f - v_ref_s * np.cross(x_h_oe, k_oe)
#     return x_h_oe

if __name__ == "__main__":
    alpha = 1
    beta = 1
    
    args = get_args()
    # args.robot = "heron"
    args.robot = "ur5e"
    
    args.pinocchio_only=True
    args.visualizer=True
    args.plotter = False
    # (robot='heron', simulation=False, robot_ip='192.168.1.102', pinocchio_only=True, ctrl_freq=500, fast_simulation=False, visualizer=True, plotter=True, gripper='none', max_iterations=100000, speed_slider=1.0, start_from_current_pose=False, acceleration=0.3, max_qd=0.5, debug_prints=False, save_log=False, save_dir='./data', run_name='latest_run', index_runs=False, past_window_size=5, controller_speed_scaling=1.0, contact_detecting_force=2.8, minimum_detectable_force_norm=3.0, visualize_collision_approximation=True, goal_error=0.01, tikhonov_damp=0.001, clik_controller='dampedPseudoinverse', alpha=0.01, beta=0.01, max_init_clik_iterations=10000, max_running_clik_iterations=1000, viz_test_path=False, randomly_generate_goal=False
    args.simulation = True 
    robot = RobotManager(args)
        
   
    Adaptive_controller = Adaptive_controller_manager(robot)
    
    # move to a proper position for initialization
    # translation = np.array([0.3,-0.3,0.5])
    # theta = np.radians(45)
    # rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [-np.sin(theta), -np.cos(theta), 0], [0, 0, -1]])
    # Mgoal = pin.SE3(rotation, translation)
    Mgoal = getRandomlyGeneratedGoal(args)
    # moveL(args, robot, Mgoal)
    print('moveL done')
    
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

from smc.robots.implementations.heron import RealHeronRobotManager, GazeboHeronRobotManager
from smc.control.control_loop_manager import ControlLoopManager
from smc import getMinimalArgParser
from smc.robots.abstract_robotmanager import AbstractRobotManager

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
# from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
# from abb_python_utilities.names import get_rosified_name
from rclpy.callback_groups import ReentrantCallbackGroup
from smc.control.cartesian_space import getClikArgs
from smc.control.optimal_control.util import get_OCP_args
# from smc.path_generation.planner import getPlanningArgs
import time
import numpy as np
import argparse
import pinocchio as pin

def get_args():
    parser = getMinimalArgParser()
    parser.description = "Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!"
    parser = getClikArgs(parser)
    parser = get_OCP_args(parser)
    # parser = getPlanningArgs(parser)
    parser.add_argument(
        "--ros-namespace",
        type=str,
        default="maskinn",
        help="you MUST put in ros namespace you're using",
    )
    parser.add_argument(
        "--unreal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="publish or not",
    )
    parser.add_argument(
        "--draw-new",
        action=argparse.BooleanOptionalAction,
        help="are you drawing a new path or reusing the previous one",
        default=False,
    )
    parser.add_argument(
        "--map-width",
        type=float,
        help="width of the map in meters (x-axis) - only used for drawing of the path",
        default=3.0,
    )
    parser.add_argument(
        "--map-height",
        type=float,
        help="height of the map in meters (y-axis) - only used for drawing of the path",
        default=3.0,
    )
    parser.add_argument(
        "--sim",
        action=argparse.BooleanOptionalAction,
        help="if in sim you need to set sim time, otherwise not",
        default=True,
    )
    # TODO: move elsewhere
    parser.add_argument(
        "--handlebar-height",
        type=float,
        default=0.5,
        help="heigh of handlebar of the cart to be pulled",
    )
    parser.add_argument(
        "--base-to-handlebar-preferred-distance",
        type=float,
        default=0.5,
        help="prefered path arclength from mobile base position to handlebar",
    )
    parser.add_argument(
        "--planner",
        action=argparse.BooleanOptionalAction,
        help="if on, you're in a pre-set map and a planner produce a plan to navigate. if off, you draw the path to be followed",
        default=True,
    )
    # parser.add_argument('--ros-args', action='extend', nargs="+")
    # parser.add_argument('-r',  action='extend', nargs="+")
    parser.add_argument("--ros-args", action="append", nargs="*")
    parser.add_argument("-r", action="append", nargs="*")
    parser.add_argument("-p", action="append", nargs="*")
    args = parser.parse_args()
    args.robot = "myumi"
    args.plotter = False
    # args.plotter = True
    args.real = True
    args.ctrl_freq = 50
    # NOTE: does not work due to ctrl-c being overriden
    args.save_log = True
    args.run_name = "nav_pls"
    args.index_runs = True
    return args


class SMCHeronNode(Node):
    def __init__(
        self,
        args,
        robot: RealHeronRobotManager,
        # robot: rosSimulatedHeronRobotManager,
        modes_and_loops: list[
            tuple[AbstractRobotManager.control_mode, ControlLoopManager]
        ],
    ):
        super().__init__("SMCHeronNode")
        if args.sim:
            self.set_parameters(
                [
                    rclpy.parameter.Parameter(
                        "use_sim_time", rclpy.Parameter.Type.BOOL, True
                    )
                ]
            )
            self.wait_for_sim_time()
        self.robot = robot
        mode, self.loop_manager = modes_and_loops.pop(0)
        self.robot.mode = mode
        self.modes_and_loops = modes_and_loops
        self.args = args
        # give me the latest thing even if there wasn't an update
        # qos_prof = rclpy.qos.QoSProfile(
        #    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
        #    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        #    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        #    depth=1,
        # )

        self._cb = ReentrantCallbackGroup()

        self._ns = args.ros_namespace
        # self._ns = get_rosified_name(self.get_namespace())

        self.get_logger().info(
            f"### Starting smc heron node example under namespace {self._ns}"
        )

        self._dt = 1 / self.args.ctrl_freq

        self._pub_timer = self.create_timer(self._dt, self.send_cmd)
        self._receive_arm_q_timer = self.create_timer(self._dt, self.receive_arm_q)
        self.current_iteration = 0

        ########################################################
        # connect to smc
        ###########################################################

        # self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos_prof)

        # this qos is incompatible
        # self.sub_joint_states = self.create_subscription(JointState, f"{self._ns}/joint_states", self.callback_arms_state, qos_prof)

        qos_prof2 = rclpy.qos.QoSProfile(
            #    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            #    durability = rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            #    history = rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._cmd_vel_pub = self.create_publisher(
            Twist, "/mir/cmd_vel", 1
        )
        # self.get_logger().info(f"{self._ns}/platform/joints_cmd")
        # self.robot.set_publisher_joints_cmd(self._cmd_pub)
        if self.args.robot == "yummi":
            self.sub_base_odom = self.create_subscription(
                Odometry, f"{self._ns}/platform/odometry", self.callback_base_odom, 1
            )
        if self.args.robot == "heron":
            self.sub_base_odom = self.create_subscription(
                Odometry, f"/mir/odom", self.callback_base_odom, 1
            )
            self.get_logger().info(
                    "subscription for odom created" 
                )
        self.odom_initialized = False
        self.init_odom = np.zeros(3)

    ##########
    # base is a twist which is constructed from the first 3 msg.velocities
    # 0 -> twist.linear.x
    # 1 -> twist.linear.y
    # 2 -> twist.angular.z
    # left arm indeces are 15-21 (last included)
    # right arm indeces are 22-28 (last included)

    def send_cmd(self):
        # self.get_logger().info("TIMER_CMD")
        self.current_iteration += 1
        breakFlag = self.loop_manager.run_one_iter(self.loop_manager.current_iteration)
        if breakFlag:
            if len(self.modes_and_loops) > 0:
                time.sleep(2)
                mode, self.loop_manager = self.modes_and_loops.pop(0)
                self.robot.mode = mode
                self.get_logger().info(
                    "about to run: " + str(self.loop_manager.controlLoop.func.__name__)
                )
            else:
                self.robot._v_cmd[:] = 0.0
                self.get_logger().info(
                    "Task finished! Set v_cmd = 0")
                self.robot.stopRobot()

        if not self.odom_initialized:
            self.get_logger().info(
                "odom intialized, hence not publishing anything!"
            )

        # self.get_logger().info("current iteration: " + str(self.current_iteration))
        # self.get_logger().info(str(self.robot._v_cmd))
        if not np.isnan(self.robot._v_cmd).any():
            twist_msg = Twist()
            # twist_msg.header.stamp = Time().to_msg()

            # TEST
            # msg.velocity[0] = (
            #    np.sin(self.loop_manager.current_iteration / (self.args.ctrl_freq * 2))
            #    / 6
            # )
            # msg.velocity[1] = (
            #    np.sin(self.loop_manager.current_iteration / (self.args.ctrl_freq * 2))
            #    / 6
            # )
            # msg.velocity[2] = (
            #    -1
            #    * np.sin(
            #        self.loop_manager.current_iteration / (self.args.ctrl_freq * 2)
            #    )
            #    / 6
            # )
            
            # REAL
            twist_msg.linear.x = self.robot._v_cmd[0] 
            twist_msg.linear.y = self.robot._v_cmd[1] 
            twist_msg.angular.z = self.robot._v_cmd[2]
            # twist_msg.angular.z = 0.1
            
            # self.get_logger().info(str(self.robot._q))
            ## TODO slower
            self._cmd_vel_pub.publish(twist_msg)
            # send v_cmd to ur5e
            # self.get_logger().info(
            #     str(self.robot._v_cmd)
            # )
            self.robot._rtde_control.speedJ(self.robot._v_cmd[3:], self.robot._acceleration, self.robot._dt)
            
    def callback_base_odom(self, msg: Odometry):
        # self.robot._v[0] = msg.twist.twist.linear.x
        # self.robot._v[1] = msg.twist.twist.linear.y
        ## TODO: check that z can be used as cos(theta) and w as sin(theta)
        ## (they could be defined some other way or theta could be offset of something)
        # self.robot._v[2] = msg.twist.twist.angular.z
        # self.robot._v[3] = 0  # for consistency
        
        # Marko's original code, there is something wrong
        # costh2 = msg.pose.pose.orientation.w
        # sinth2 = np.linalg.norm(
        #     [
        #         msg.pose.pose.orientation.x,
        #         msg.pose.pose.orientation.y,
        #         msg.pose.pose.orientation.z,
        #     ]
        # )
        # th = 2 * np.arctan2(sinth2, costh2)
        
        # My version
        q = msg.pose.pose.orientation
        th = -2 * np.arctan2(q.z, q.w)
        
        if not self.odom_initialized:
            self.init_odom[0] = msg.pose.pose.position.x
            self.init_odom[1] = msg.pose.pose.position.y
            self.init_odom[2] = th
            self.odom_initialized = True
            self.get_logger().info(str(self.init_odom))
        if (self.odom_initialized) or self.current_iteration < 50:
            T_odom = np.zeros((3, 3))
            T_odom[0, 0] = np.cos(self.init_odom[2])
            T_odom[0, 1] = -1 * np.sin(self.init_odom[2])
            T_odom[0, 2] = self.init_odom[0]
            T_odom[1, 0] = np.sin(self.init_odom[2])
            T_odom[1, 1] = np.cos(self.init_odom[2])
            T_odom[1, 2] = self.init_odom[1]
            T_odom[2, 2] = 1.0
            p_odom = np.array(
                [
                    msg.pose.pose.position.x - self.init_odom[0],
                    msg.pose.pose.position.y - self.init_odom[1],
                ]
            )
            # T_inv_odom = np.zeros((3, 3))
            # T_inv_odom[:2, :2] = T_odom[:2, :2].T
            # T_inv_odom[:2, 2] = (-1 * T_odom[:2, :2].T) @ T_odom[:2, 2]
            # T_inv_odom[2, 2] = 1.0
            # p_ctrl = T_inv_odom @ p_odom
            # p_ctrl = (T_odom @ p_odom)[:2]
            p_ctrl = T_odom[:2, :2] @ p_odom
            self.robot._q[0] = p_ctrl[0]
            self.robot._q[1] = p_ctrl[1]
            self.robot._q[2] = np.cos(self.init_odom[2] - th)
            self.robot._q[3] = np.sin(self.init_odom[2] - th)
            # self.get_logger().info(str(self.robot._q))
            # self.get_logger().info(str(th))
            # self.get_logger().info(str(self.init_odom))
        # self.get_logger().info("CALLBACK_ODOM")
        # self.get_logger().info(str(self.robot._q[:4]))
        # self.get_logger().info(str(self.init_odom))

    def receive_arm_q(self):
        q = self.robot._rtde_receive.getActualQ()
        self.robot._q[4:] = np.array(q)

    def wait_for_sim_time(self):
        """Wait for the /clock topic to start publishing."""
        self.get_logger().info("Waiting for simulated time to be active...")
        while not self.get_clock().now().nanoseconds > 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Simulated time is now active!")



class GazeboSMCHeronNode(Node):
    def __init__(
        self,
        args,
        robot: GazeboHeronRobotManager,
        # robot: rosSimulatedHeronRobotManager,
        modes_and_loops: list[
            tuple[AbstractRobotManager.control_mode, ControlLoopManager]
        ],
    ):
        super().__init__("SMCHeronNode")
        if args.sim:
            self.set_parameters(
                [
                    rclpy.parameter.Parameter(
                        "use_sim_time", rclpy.Parameter.Type.BOOL, True
                    )
                ]
            )
            self.wait_for_sim_time()
        self.robot = robot
        mode, self.loop_manager = modes_and_loops.pop(0)
        self.robot.mode = mode
        self.modes_and_loops = modes_and_loops
        self.args = args
        # give me the latest thing even if there wasn't an update
        # qos_prof = rclpy.qos.QoSProfile(
        #    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
        #    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        #    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        #    depth=1,
        # )

        self._cb = ReentrantCallbackGroup()

        self._ns = args.ros_namespace
        # self._ns = get_rosified_name(self.get_namespace())

        self.get_logger().info(
            f"### Starting smc heron node example under namespace {self._ns}"
        )

        self._dt = 1 / self.args.ctrl_freq

        self._pub_timer = self.create_timer(self._dt, self.send_cmd)
        self._receive_arm_q_timer = self.create_timer(self._dt, self.receive_arm_q)
        self.current_iteration = 0

        ########################################################
        # connect to smc
        ###########################################################

        # self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos_prof)

        # this qos is incompatible
        # self.sub_joint_states = self.create_subscription(JointState, f"{self._ns}/joint_states", self.callback_arms_state, qos_prof)

        qos_prof2 = rclpy.qos.QoSProfile(
            #    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            #    durability = rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            #    history = rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._cmd_vel_pub = self.create_publisher(
            Twist, "/cmd_vel", 1
        )
        # self.get_logger().info(f"{self._ns}/platform/joints_cmd")
        # self.robot.set_publisher_joints_cmd(self._cmd_pub)
        if self.args.robot == "yummi":
            self.sub_base_odom = self.create_subscription(
                Odometry, f"{self._ns}/platform/odometry", self.callback_base_odom, 1
            )
        if self.args.robot == "heron":
            self.sub_base_odom = self.create_subscription(
                Odometry, f"/odom", self.callback_base_odom, 1
            )
            self.get_logger().info(
                    "subscription for odom created" 
                )
        self.odom_initialized = False
        self.init_odom = np.zeros(3)

    ##########
    # base is a twist which is constructed from the first 3 msg.velocities
    # 0 -> twist.linear.x
    # 1 -> twist.linear.y
    # 2 -> twist.angular.z
    # left arm indeces are 15-21 (last included)
    # right arm indeces are 22-28 (last included)

    def send_cmd(self):
        # self.get_logger().info("TIMER_CMD")
        self.current_iteration += 1
        breakFlag = self.loop_manager.run_one_iter(self.loop_manager.current_iteration)
        if breakFlag:
            if len(self.modes_and_loops) > 0:
                mode, self.loop_manager = self.modes_and_loops.pop(0)
                self.robot.mode = mode
                self.get_logger().info(
                    "about to run: " + str(self.loop_manager.controlLoop.func.__name__)
                )
            else:
                self.robot._v_cmd[:] = 0.0
                self.get_logger().info(
            "Task finished! Set v_cmd = 0"
        )
        if not self.odom_initialized:
            self.get_logger().info(
                "odom not initialized, hence not publishing anything!"
            )

        # self.get_logger().info("current iteration: " + str(self.current_iteration))
        # self.get_logger().info(str(self.robot._v_cmd))
        if self.args.unreal:
            twist_msg = Twist()
            # twist_msg.header.stamp = Time().to_msg()

            # TEST
            # msg.velocity[0] = (
            #    np.sin(self.loop_manager.current_iteration / (self.args.ctrl_freq * 2))
            #    / 6
            # )
            # msg.velocity[1] = (
            #    np.sin(self.loop_manager.current_iteration / (self.args.ctrl_freq * 2))
            #    / 6
            # )
            # msg.velocity[2] = (
            #    -1
            #    * np.sin(
            #        self.loop_manager.current_iteration / (self.args.ctrl_freq * 2)
            #    )
            #    / 6
            # )
            # REAL
            twist_msg.linear.x = self.robot._v_cmd[0] 
            twist_msg.linear.y = self.robot._v_cmd[1] 
            twist_msg.angular.z = self.robot._v_cmd[2]
            # self.get_logger().info(str(self.robot._q))
            ## TODO slower
            self._cmd_vel_pub.publish(twist_msg)
            # send v_cmd to ur5e
            self.robot._q = pin.integrate(self.robot.model, self.robot._q, self.robot._v_cmd * self.robot._dt)
             
    def callback_base_odom(self, msg: Odometry):
        # self.robot._v[0] = msg.twist.twist.linear.x
        # self.robot._v[1] = msg.twist.twist.linear.y
        ## TODO: check that z can be used as cos(theta) and w as sin(theta)
        ## (they could be defined some other way or theta could be offset of something)
        # self.robot._v[2] = msg.twist.twist.angular.z
        # self.robot._v[3] = 0  # for consistency
        
        # Marko's original code, there is something wrong
        # costh2 = msg.pose.pose.orientation.w
        # sinth2 = np.linalg.norm(
        #     [
        #         msg.pose.pose.orientation.x,
        #         msg.pose.pose.orientation.y,
        #         msg.pose.pose.orientation.z,
        #     ]
        # )
        # th = 2 * np.arctan2(sinth2, costh2)
        
        # My version
        q = msg.pose.pose.orientation
        th = -2 * np.arctan2(q.z, q.w)
        
        if not self.odom_initialized:
            self.init_odom[0] = msg.pose.pose.position.x
            self.init_odom[1] = msg.pose.pose.position.y
            self.init_odom[2] = th
            self.odom_initialized = True
            self.get_logger().info(str(self.init_odom))
        if (self.args.unreal and self.odom_initialized) or self.current_iteration < 50:
            T_odom = np.zeros((3, 3))
            T_odom[0, 0] = np.cos(self.init_odom[2])
            T_odom[0, 1] = -1 * np.sin(self.init_odom[2])
            T_odom[0, 2] = self.init_odom[0]
            T_odom[1, 0] = np.sin(self.init_odom[2])
            T_odom[1, 1] = np.cos(self.init_odom[2])
            T_odom[1, 2] = self.init_odom[1]
            T_odom[2, 2] = 1.0
            p_odom = np.array(
                [
                    msg.pose.pose.position.x - self.init_odom[0],
                    msg.pose.pose.position.y - self.init_odom[1],
                ]
            )
            # T_inv_odom = np.zeros((3, 3))
            # T_inv_odom[:2, :2] = T_odom[:2, :2].T
            # T_inv_odom[:2, 2] = (-1 * T_odom[:2, :2].T) @ T_odom[:2, 2]
            # T_inv_odom[2, 2] = 1.0
            # p_ctrl = T_inv_odom @ p_odom
            # p_ctrl = (T_odom @ p_odom)[:2]
            p_ctrl = T_odom[:2, :2] @ p_odom
            self.robot._q[0] = p_ctrl[0]
            self.robot._q[1] = p_ctrl[1]
            self.robot._q[2] = np.cos(self.init_odom[2] - th)
            self.robot._q[3] = np.sin(self.init_odom[2] - th)
            # self.get_logger().info(str(self.robot._q))
            # self.get_logger().info(str(th))
            # self.get_logger().info(str(self.init_odom))
        # self.get_logger().info("CALLBACK_ODOM")
        # self.get_logger().info(str(self.robot._q[:4]))
        # self.get_logger().info(str(self.init_odom))

    def receive_arm_q(self):
        pass

    def wait_for_sim_time(self):
        """Wait for the /clock topic to start publishing."""
        self.get_logger().info("Waiting for simulated time to be active...")
        while not self.get_clock().now().nanoseconds > 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Simulated time is now active!")

# PYTHON_ARGCOMPLETE_OK
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from abb_python_utilities.names import get_rosified_name

import numpy as np
import argcomplete, argparse
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
from ur_simple_control.clik.clik import getClikArgs, getClikController, controlLoopClik, moveL, compliantMoveL, controlLoopCompliantClik, invKinmQP, moveLDualArm
from ur_simple_control.optimal_control.get_ocp_args import get_OCP_args
from ur_simple_control.optimal_control.crocoddyl_mpc import CrocoIKMPC
import pinocchio as pin


###########################################################
# this is how to get the positions  ros2 topic echo /maskinn/auxiliary/joint_states_merged
# this is how to send commands: /maskinn/auxiliary/robot_description
#self._cmd_pub = self.create_publisher(JointState, f"{self._ns}/joints_cmd", 1)
#############################################33


def get_args():
    parser = getMinimalArgParser()
    parser.description = 'Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!'
    parser = getClikArgs(parser)
    parser = get_OCP_args(parser)
    parser.add_argument('--ros-namespace', type=str, default="maskinn", help="you MUST put in ros namespace you're using")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

class ROSCommHandlerMYumi(Node):

    def __init__(self, args, robot_manager : RobotManager, loop_manager : ControlLoopManager):
        super().__init__('ROSCommHandlerMYumi')
        
        # does not work
        #self._ns = get_rosified_name(self.get_namespace())
        # so we pass it as an argument
        self._ns = args.ros_namespace + "/"
        self.get_logger().info(f"### Starting dummy example under namespace {self._ns}")
        
        #self.pub_vel_base = self.create_publisher(Twist, f"{self._ns}platform/base_command", 1)
        #self.pub_vel_base = self.create_publisher(Twist, f"{self._ns}platform/command_limiter_node/base/cmd_vel_in_navigation", 1)
        #self.pub_vel_left = self.create_publisher(JointState, f"{self._ns}platform/left_arm_command", 1)
        #self.pub_vel_right = self.create_publisher(JointState, f"{self._ns}platform/right_arm_command", 1)

        robot_manager.set_publisher_vel_base(self.pub_vel_base)
        robot_manager.set_publisher_vel_left(self.pub_vel_left)
        robot_manager.set_publisher_vel_right(self.pub_vel_right)
        
        self.robot_manager = robot_manager
        self.loop_manager = loop_manager
        self.args = args
        
        # give me the latest thing even if there wasn't an update
        qos_prof = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability = rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history = rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth = 1)

        # MASSIVE TODO: you want to be subsribed to a proper localization thing,
        # specifically to output from robot_localization
        self.sub_base_odom = self.create_subscription(Odometry, f"{self._ns}platform/odometry", self.callback_base_odom, qos_prof)
        # self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos_prof)
        self.sub_arms_state = self.create_subscription(JointState, f"{self._ns}platform/joint_states", self.callback_arms_state, qos_prof)

        self.names_left = {f"{self._ns}_yumi_robl_joint_{i+1}": i for i in range(7)}
        self.state_left = JointState(name=list(sorted(self.names_left.keys())),
                                     position=[0] * 7,
                                     velocity=[0] * 7,
                                     effort=[0] * 7)
        self.names_right = {f"{self._ns}_yumi_robr_joint_{i+1}": i for i in range(7)}
        self.state_right = JointState(name=list(sorted(self.names_right.keys())),
                                      position=[0] * 7,
                                      velocity=[0] * 7,
                                      effort=[0] * 7)
        ### somewhat useless 
        # f"{self._ns}/platform/left_arm_pose" : PoseStamped
        # f"{self._ns}/platform/right_arm_pose" : PoseStamped
        ### for updating the obstacles
        # f"{self._ns}/camera/points2" : PointCloud2
        # f"{self._ns}/sensors/front/lidar/scan" : LaserScan
        # f"{self._ns}/sensors/read/lidar/scan" : LaserScan
        
        ctrl_rate = 250  # hz 
        self._pub_timer = self.create_timer(1/ctrl_rate, self.send_commands)

    # ONE TRILLION PERCENT YOU NEED TO INTEGRATE/USE ODOM IN-BETWEEN THESE
    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_manager.q[0] = msg.pose.pose.position.x
        self.robot_manager.q[1] = msg.pose.pose.position.y
        # TODO: check that z can be used as cos(theta) and w as sin(theta)
        # (they could be defined some other way or theta could be offset of something)
        # ONE TRILLION PERCENT THIS ISN'T CORRECT
        # TODO: generate a rotation matrix from this quarternion with pinocchio,
        # and give this is the value of the x or the y axis
        costh2 = msg.pose.pose.orientation.w
        sinth2 = np.linalg.norm([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        th = 2*np.arctan2(sinth2, costh2)
        self.robot_manager.q[2] = np.cos(th)
        self.robot_manager.q[3] = np.sin(th)
        print("received /amcl_pose")

    # HIGHLY LIKELY THAT THIS IS NOT THE VELOCITY OF THE BASE FRAME
    def callback_base_odom(self, msg : Odometry):
        self.robot_manager.v_q[0] = msg.twist.twist.linear.x
        self.robot_manager.v_q[1] = msg.twist.twist.linear.y
        # TODO: check that z can be used as cos(theta) and w as sin(theta)
        # (they could be defined some other way or theta could be offset of something)
        self.robot_manager.v_q[2] = msg.twist.twist.angular.z
        self.robot_manager.v_q[3] = 0  # for consistency
        print("received /odom")

    def callback_arms_state(self, msg : JointState):
        for name, pos, vel, eff in zip(msg.name, msg.position, msg.velocity, msg.effort):
            if name in self.names_left.keys():
                i = self.names_left[name]
                self.state_left.position[i] = pos
                self.state_left.velocity[i] = vel
                self.state_left.effort[i] = eff
            elif name in self.names_right.keys():
                i = self.names_right[name]
                self.state_right.position[i] = pos
                self.state_right.velocity[i] = vel
                self.state_right.effort[i] = eff
        for i in range(7):
            self.robot_manager.q[4+i] = self.state_left.position[i]
            self.robot_manager.q[4+i+7] = self.state_right.position[i]
            self.robot_manager.v_q[4+i] = self.state_left.velocity[i]
            self.robot_manager.v_q[4+i+7] = self.state_right.velocity[i]

    def send_commands(self):
        breakFlag = self.loop_manager.run_one_iter(0)
        
        # msg = Twist()
        # msg.linear.x = 0
        # self.publisher_vel_base.publish(msg)
        #
        # msg = JointState()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.velocity = [0, 0, 0, 0, 0, 0, 0]
        # self.publisher_vel_left.publish(msg)
        #
        # msg = JointState()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.velocity = [0, 0, 0, 0, 0, 0, 0]
        # self.publisher_vel_right.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    
    args = get_args()
    robot = RobotManager(args)
    goal = robot.defineGoalPointCLI()
    if robot.robot_name == "yumi":
        goal.rotation = np.eye(3)
        goal_transform = pin.SE3.Identity()
        goal_transform.translation[1] = 0.1
        goal_left = goal_transform.act(goal)
        goal_left = goal_transform.inverse().act(goal)
#        goal = (goal_left, goal_right)
    #loop_manager = CrocoIKMPC(args, robot, goal, run=False)
    loop_manager = moveLDualArm(args, robot, goal, goal_transform, run=False)
    
    executor = MultiThreadedExecutor()
    ros_comm_handler = ROSCommHandlerMYumi(args, robot, loop_manager)
    executor.add_node(ros_comm_handler)
    executor.spin()
    
    if args.save_log:
        robot.log_manager.saveLog()
        robot.log_manager.plotAllControlLoops()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ros_comm_handler.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

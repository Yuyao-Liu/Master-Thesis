# PYTHON_ARGCOMPLETE_OK
import rclpy
from rclpy.node import Node
from geometry_msgs import msg 
from nav_msgs.msg import Odometry
from rclpy import wait_for_message
import pinocchio as pin
import argcomplete, argparse
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
from ur_simple_control.clik.clik import getClikArgs, getClikController, controlLoopClik, moveL, compliantMoveL, controlLoopCompliantClik, invKinmQP
from ur_simple_control.optimal_control.get_ocp_args import get_OCP_args
from ur_simple_control.optimal_control.crocoddyl_mpc import CrocoIKMPC
import threading


def get_args():
    parser = getMinimalArgParser()
    parser.description = 'Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!'
    parser = getClikArgs(parser)
    parser = get_OCP_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args

class ROSCommHandlerHeron(Node):

    def __init__(self, args, robot_manager : RobotManager, loop_manager : ControlLoopManager):
        super().__init__('ROSCommHandlerHeron')
        
        self.publisher_vel_base = self.create_publisher(msg.Twist, '/cmd_vel', 5)
        print("created publisher")
        robot_manager.set_publisher_vel_base(self.publisher_vel_base)
        
        qos_prof = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability = rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history = rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth = 1,
        )
# MASSIVE TODO: you want to be subsribed to a proper localization thing,
# specifically to output from robot_localization
        self.subscription_amcl = self.create_subscription(msg.PoseWithCovarianceStamped, 
                                            '/amcl_pose', self.pose_callback, qos_prof)
        self.subscription_odom = self.create_subscription(Odometry, 
                                            '/odom', self.odom_callback, qos_prof)
        #wait_for_message.wait_for_message(msg.PoseWithCovarianceStamped, self, '/amcl_pose')
        #print("subscribed to /amcl_pose")
        self.subscription_amcl  # prevent unused variable warning
        self.subscription_odom  # prevent unused variable warning
        self.robot_manager = robot_manager
        self.loop_manager = loop_manager
        self.args = args

    # ONE TRILLION PERCENT YOU NEED TO INTEGRATE/USE ODOM IN-BETWEEN THESE
    def pose_callback(self, mesg):
        self.robot_manager.q[0] = mesg.pose.pose.position.x
        self.robot_manager.q[1] = mesg.pose.pose.position.y
        # TODO: check that z can be used as cos(theta) and w as sin(theta)
        # (they could be defined some other way or theta could be offset of something)
    # ONE TRILLION PERCENT THIS ISN'T CORRECT
    # TODO: generate a rotation matrix from this quarternion with pinocchio,
    # and give this is the value of the x or the y axis 
        self.robot_manager.q[2] = mesg.pose.pose.orientation.z
        self.robot_manager.q[3] = mesg.pose.pose.orientation.w
        
        #vel_cmd = msg.Twist()
        print("received new amcl")
        #self.publisher_vel_base.publish(vel_cmd)

    # HIGHLY LIKELY THAT THIS IS NOT THE THE VELOCITY OF THE BASE FRAME
    def odom_callback(self, mesg : Odometry):
        self.robot_manager.v_q[0] = mesg.twist.twist.linear.x
        self.robot_manager.v_q[1] = mesg.twist.twist.linear.y
        # TODO: check that z can be used as cos(theta) and w as sin(theta)
        # (they could be defined some other way or theta could be offset of something)
        self.robot_manager.v_q[2] = mesg.twist.twist.angular.z
        
        #vel_cmd = msg.Twist()
        print("received new amcl")
        #self.publisher_vel_base.publish(vel_cmd)



if __name__ == '__main__':
    rclpy.init()
    args = get_args()
    robot = RobotManager(args)
    goal = robot.defineGoalPointCLI()
    #goal = pin.SE3.Identity()
    #loop_manager = compliantMoveL(args, robot, goal, run=False)
    loop_manager = CrocoIKMPC(args, robot, goal, run=False)
    ros_comm_handler = ROSCommHandlerHeron(args, robot, loop_manager)
    
    thread = threading.Thread(target=rclpy.spin, args=(ros_comm_handler,), daemon=True)
    thread.start()

    rate = ros_comm_handler.create_rate(250)
    while rclpy.ok():
        #msgg = msg.Twist()
        #print("publisihng")
        #ros_comm_handler.publisher_vel_base.publish(msgg)
        breakFlag = ros_comm_handler.loop_manager.run_one_iter(0)
        rate.sleep()
        


#    if not args.pinocchio_only:
#        robot.stopRobot()
#
#    if args.visualize_manipulator:
#        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()
        robot.log_manager.plotAllControlLoops()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ros_comm_handler.destroy_node()
    rclpy.shutdown()





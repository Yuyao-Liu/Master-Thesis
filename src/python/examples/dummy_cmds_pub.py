#!/usr/bin/env python3
#
# Copyright (c) 2024, ABB Schweiz AG
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from abb_python_utilities.names import get_rosified_name

import numpy as np
import argparse
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
from ur_simple_control.clik.clik import getClikArgs, getClikController, controlLoopClik, moveL, compliantMoveL, controlLoopCompliantClik, invKinmQP, moveLDualArm
from ur_simple_control.optimal_control.get_ocp_args import get_OCP_args
from ur_simple_control.optimal_control.crocoddyl_mpc import CrocoIKMPC
import pinocchio as pin

import numpy as np

import rclpy
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from abb_python_utilities.names import get_rosified_name


# you will need to manually set everything here :(
def get_args():
    parser = getMinimalArgParser()
    parser.description = 'Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!'
    parser = getClikArgs(parser)
    parser = get_OCP_args(parser)
    parser.add_argument('--ros-namespace', type=str, default="maskinn", help="you MUST put in ros namespace you're using")
    #parser.add_argument('--ros-args', action='extend', nargs="+")
    #parser.add_argument('-r',  action='extend', nargs="+")
    parser.add_argument('--ros-args', action='append', nargs='*')
    parser.add_argument('-r',  action='append', nargs='*')
    parser.add_argument('-p',  action='append', nargs='*')
    args = parser.parse_args()
    args.robot = "yumi"
    args.save_log = True
    args.real_time_plotting = False
    return args


class DummyNode(Node):
    def __init__(self, args, robot_manager : RobotManager, loop_manager : ControlLoopManager):
        super().__init__("dummy_cmds_pub_node")

        self._cb = ReentrantCallbackGroup()

        self._ns = get_rosified_name(self.get_namespace())

        self._cmd_pub = self.create_publisher(JointState, f"{self._ns}/joints_cmd", 1)

        self.get_logger().info(f"### Starting dummy example under namespace {self._ns}")

        self.empty_msg = JointState()
        for i in range(29):
            self.empty_msg.velocity.append(0.0)

        self._T = 1.0
        self._A = 0.2
        self._t = 0.0
        self._dt = 0.004

        self._pub_timer = self.create_timer(self._dt, self.send_cmd)

########################################################
# connect to smc
###########################################################

        robot_manager.set_publisher_joints_cmd(self._cmd_pub)
        self.robot_manager = robot_manager
        self.loop_manager = loop_manager
        self.args = args
        # give me the latest thing even if there wasn't an update
        qos_prof = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability = rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history = rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth = 1)

#        self.sub_base_odom = self.create_subscription(Odometry, f"{self._ns}platform/odometry", self.callback_base_odom, qos_prof)
        # self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos_prof)

        # this qos is incompatible
        #self.sub_joint_states = self.create_subscription(JointState, f"{self._ns}/joint_states", self.callback_arms_state, qos_prof)

        qos_prof2 = rclpy.qos.QoSProfile(
        #    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
        #    durability = rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        #    history = rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth = 1)
        self.sub_joint_states = self.create_subscription(JointState, f"{self._ns}/joint_states", self.callback_arms_state, qos_prof2)



##########
# base is a twist which is constructed from the first 3 msg.velocities
# 0 -> twist.linear.x
# 1 -> twist.linear.y 
# 2 -> twist.angular.z
# left arm indeces are 15-21 (last included)
# right arm indeces are 22-28 (last included)

    def send_cmd(self):
        breakFlag = self.loop_manager.run_one_iter(0)
        #msg = self.empty_msg
        #msg.header.stamp = self.get_clock().now().to_msg()
        #msg.velocity[0] = self._A * np.sin(2 * np.pi / self._T * self._t)
        #msg.velocity[1] = self._A * np.sin(2 * np.pi / self._T * self._t)
        #msg.velocity[2] = self._A * np.sin(2 * np.pi / self._T * self._t)
        #self._cmd_pub.publish(msg)
        #self._t += self._dt


    def callback_base_odom(self, msg : Odometry):
        self.robot_manager.v_q[0] = msg.twist.twist.linear.x
        self.robot_manager.v_q[1] = msg.twist.twist.linear.y
        # TODO: check that z can be used as cos(theta) and w as sin(theta)
        # (they could be defined some other way or theta could be offset of something)
        self.robot_manager.v_q[2] = msg.twist.twist.angular.z
        self.robot_manager.v_q[3] = 0  # for consistency
        print("received /odom")

    def callback_arms_state(self, msg : JointState):
        self.robot_manager.q[0] = 0.0
        self.robot_manager.q[1] = 0.0
        self.robot_manager.q[2] = 1.0
        self.robot_manager.q[3] = 0.0
        self.robot_manager.q[-14:-7] = msg.position[-14:-7]
        self.robot_manager.q[-7:] = msg.position[-7:]
        self.get_logger().info('jebem ti mamu')


#def main(args=None):
def main(args=None):
    # evil and makes args unusable but what can you do
    args_smc = get_args()
    #print(args_smc)
    rclpy.init(args=args)

    robot = RobotManager(args_smc)
# you can't do terminal input with ros
    #goal = robot.defineGoalPointCLI()
    goal = pin.SE3.Identity()
    goal.translation = np.array([0.3,0.0,0.5])
    if robot.robot_name == "yumi":
        goal.rotation = np.eye(3)
        goal_transform = pin.SE3.Identity()
        goal_transform.translation[1] = 0.1
        goal_left = goal_transform.act(goal)
        goal_left = goal_transform.inverse().act(goal)
#        goal = (goal_left, goal_right)
    #loop_manager = CrocoIKMPC(args, robot, goal, run=False)
    loop_manager = moveLDualArm(args_smc, robot, goal, goal_transform, run=False)


    executor = MultiThreadedExecutor()
    node = DummyNode(args, robot, loop_manager)
    executor.add_node(node)
    executor.spin()


if __name__ == "__main__":
    main()

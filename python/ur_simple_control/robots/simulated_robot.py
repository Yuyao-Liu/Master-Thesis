from ur_simple_control.robots.robotmanager_abstract import RobotManagerAbstract
from ur_simple_control.util.get_model import *


class RobotManagerSimulated(RobotManagerAbstract):
    def __init__(self, args):
        # TODO: idk where i pass args
        super(RobotManagerAbstract, self).__init__(args)
        if self.robot_name == "ur5e":
            self.model, self.collision_model, self.visual_model, self.data = get_model()
        if self.robot_name == "heron":
            self.model, self.collision_model, self.visual_model, self.data = (
                heron_approximation()
            )
        if self.robot_name == "heronros":
            self.model, self.collision_model, self.visual_model, self.data = (
                heron_approximation()
            )
        if self.robot_name == "mirros":
            self.model, self.collision_model, self.visual_model, self.data = (
                mir_approximation()
            )
            # self.publisher_vel_base = create_publisher(msg.Twist, '/cmd_vel', 5)
            # self.publisher_vel_base = publisher_vel_base
        if self.robot_name == "gripperlessur5e":
            self.model, self.collision_model, self.visual_model, self.data = (
                getGripperlessUR5e()
            )
        if self.robot_name == "yumi":
            self.model, self.collision_model, self.visual_model, self.data = (
                get_yumi_model()

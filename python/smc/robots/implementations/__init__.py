from .mir import SimulatedMirRobotManager
from .heron import SimulatedHeronRobotManager
from .mobile_yumi import SimulatedMobileYuMiRobotManager
from .ur5e import SimulatedUR5eRobotManager
from .yumi import SimulatedYuMiRobotManager

from importlib.util import find_spec

if find_spec("rtde_control"):
    from .ur5e import RealUR5eRobotManager

if find_spec("rclpy"):
    from .yumi import RealYuMiRobotManager
    from .mobile_yumi import RealMobileYumiRobotManager

if find_spec("rclpy") and find_spec("rtde_control"):
    from .heron import RealHeronRobotManager

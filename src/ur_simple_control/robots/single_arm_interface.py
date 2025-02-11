import abs
from ur_simple_control.robots.robotmanager_abstract import RobotManagerAbstract


class SingleArmInterface(RobotManagerAbstract):
    def __init__(self):
        # idk if this is correct
        super().__init__
        self.ee_frame_id = self.model.getFrameId("tool0")

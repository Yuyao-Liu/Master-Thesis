
class RobotManagerMobileYumi(RobotManagerAbstract):
    def __init__(self, args):
        self.args = args
        self.model, self.collision_model, self.visual_model, self.data = (
            get_yumi_model()
        self.r_ee_frame_id = self.model.getFrameId("robr_joint_7")
        self.l_ee_frame_id = self.model.getFrameId("robl_joint_7")

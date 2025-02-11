class RobotManagerHeron(RobotManagerAbstract):
    def __init__(self, args):
        self.args = args
        self.model, self.collision_model, self.visual_model, self.data = (
            heron_approximation()
        )
        self.ee_frame_id = self.model.getFrameId("tool0")

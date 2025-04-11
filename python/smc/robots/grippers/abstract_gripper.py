import abc


class AbstractGripper(abc.ABC):
    """
    AbstractGripper
    ------------------
    Abstract gripper class enforcing all grippers to have the same API toward RobotManager.
    The point of this is to avoid having too many ifs in RobotManager
    which reduce its readability, while achieving the same effect
    of moving the stupid hardware bookkeeping out of sight.
    Bookkeeping here refers to various grippers using different interfaces and differently
    named functions just to do the same thing - move the gripper to a certain position
    with a certain speed and certain final gripping force.
    There are also the classic expected open(), close(), isGripping() quality-of-life functions.
    """

    # TODO: make this abstract as well,
    # but make it possible to have more arguments, or go to grippers and make keyword arguments
    # for the extra stuff on a case-by-case basis
    # def connect(self):
    #    pass

    @abc.abstractmethod
    def move(self, position, speed=None, gripping_force=None):
        pass

    @abc.abstractmethod
    def open(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


#    def setVelocity(self):
#        pass
#
#    def setGrippingForce(self):
#        pass
#

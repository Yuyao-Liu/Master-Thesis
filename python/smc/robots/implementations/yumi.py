from smc.robots.abstract_robotmanager import AbstractRealRobotManager
from smc.robots.abstract_simulated_robotmanager import AbstractSimulatedRobotManager
from smc.robots.interfaces.dual_arm_interface import DualArmInterface

from argparse import Namespace
from importlib.resources import files
from os import path
import pinocchio as pin
import numpy as np


class AbstractYuMiRobotManager(DualArmInterface):
    def __init__(self, args):
        if args.debug_prints:
            print("YuMiRobotManager init")
        self._model, self._collision_model, self._visual_model, self._data = (
            get_yumi_model()
        )
        self._l_ee_frame_id = self.model.getFrameId("robl_tool0")
        self._r_ee_frame_id = self.model.getFrameId("robr_tool0")
        self._MAX_ACCELERATION = 1.7  # const
        self._MAX_QD = 3.14  # const
        #        self._comfy_configuration = np.array(
        #            [
        #                -0.7019,
        #                0.03946,
        #                1.13817,
        #                0.40438,
        #                1.59454,
        #                0.37243,
        #                -1.3882,
        #                1.17810,
        #                -0.00055,
        #                -1.7492,
        #                0.41061,
        #                -2.0604,
        #                0.30449,
        #                1.72462,
        #            ]
        #        )
        self._comfy_configuration = np.array(
            [
                0.045,
                -0.155,
                -0.394,
                -0.617,
                -0.939,
                -0.343,
                -1.216,
                -0.374,
                -0.249,
                0.562,
                -0.520,
                0.934,
                -0.337,
                1.400,
            ]
        )

        super().__init__(args)


class SimulatedYuMiRobotManager(
    AbstractSimulatedRobotManager, AbstractYuMiRobotManager
):
    def __init__(self, args: Namespace):
        if args.debug_prints:
            print("SimulatedYuMiRobotManager init")
        super().__init__(args)


class RealYuMiRobotManager(AbstractRealRobotManager, AbstractYuMiRobotManager):
    pass


def get_yumi_model() -> (
    tuple[pin.Model, pin.GeometryModel, pin.GeometryModel, pin.Data]
):

    urdf_path_relative = files("smc.robots.robot_descriptions").joinpath("yumi.urdf")
    urdf_path_absolute = path.abspath(urdf_path_relative)
    # mesh_dir = files('smc')
    # mesh_dir_absolute = os.path.abspath(mesh_dir)
    # mesh_dir_absolute = "/home/gospodar/lund/praxis/software/ros/ros-containers/home/heron_description/MIR_robot"
    mesh_dir_absolute = None

    # this command just calls the ones below it. both are kept here
    # in case pinocchio people decide to change their api.
    # model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path_absolute, mesh_dir_absolute)
    model = pin.buildModelFromUrdf(urdf_path_absolute)
    visual_model = pin.buildGeomFromUrdf(
        model, urdf_path_absolute, pin.GeometryType.VISUAL, None, mesh_dir_absolute
    )
    collision_model = pin.buildGeomFromUrdf(
        model,
        urdf_path_absolute,
        pin.GeometryType.COLLISION,
        None,
        mesh_dir_absolute,
    )

    data = pin.Data(model)
    return model, collision_model, visual_model, data

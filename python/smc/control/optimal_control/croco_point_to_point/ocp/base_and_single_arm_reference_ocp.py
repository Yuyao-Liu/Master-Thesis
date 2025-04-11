from smc.robots.interfaces.whole_body_single_arm_interface import (
    SingleArmWholeBodyInterface,
)
from smc.control.optimal_control.croco_point_to_point.ocp.base_reference_ocp import (
    BaseIKOCP,
)
from smc.control.optimal_control.croco_point_to_point.ocp.single_arm_reference_ocp import (
    SingleArmIKOCP,
)

import numpy as np
from argparse import Namespace


class BaseAndSingleArmIKOCP(SingleArmIKOCP, BaseIKOCP):
    def __init__(
        self,
        args: Namespace,
        robot: SingleArmWholeBodyInterface,
        x0: np.ndarray,
        goal,
    ):
        super().__init__(args, robot, x0, goal)

    def constructTaskCostsValues(self):
        self.base_translation_cost_values = np.linspace(
            self.args.base_translation_cost,
            self.args.base_translation_cost
            * self.args.linearly_increasing_task_cost_factor,
            self.args.n_knots + 1,
        )
        self.ee_pose_cost_values = np.linspace(
            self.args.ee_pose_cost,
            self.args.ee_pose_cost * self.args.linearly_increasing_task_cost_factor,
            self.args.n_knots + 1,
        )

    def constructTaskObjectiveFunction(
        self,
        goal,
    ) -> None:
        T_w_eegoal, p_basegoal = goal
        super().constructTaskObjectiveFunction(T_w_eegoal)
        BaseIKOCP.constructTaskObjectiveFunction(self, p_basegoal)

    # there is nothing to update in a point-to-point task
    def updateCosts(self, data):
        pass

    def updateGoalInModels(self, goal) -> None:
        # self, T_w_eegoal: pin.SE3, p_basegoal: np.ndarray) -> None:
        T_w_eegoal, p_basegoal = goal
        super().updateGoalInModels(T_w_eegoal)
        BaseIKOCP.updateGoalInModels(self, p_basegoal)

from smc.robots.interfaces.whole_body_dual_arm_interface import (
    DualArmWholeBodyInterface,
)
from smc.control.optimal_control.croco_point_to_point.ocp.base_reference_ocp import (
    BaseIKOCP,
)
from smc.control.optimal_control.croco_point_to_point.ocp.dual_arm_reference_ocp import (
    DualArmIKOCP,
)

import numpy as np
from argparse import Namespace


class BaseAndDualArmIKOCP(DualArmIKOCP, BaseIKOCP):
    def __init__(
        self,
        args: Namespace,
        robot: DualArmWholeBodyInterface,
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
        T_w_lgoal, T_w_rgoal, p_basegoal = goal
        super().constructTaskObjectiveFunction((T_w_lgoal, T_w_rgoal))
        BaseIKOCP.constructTaskObjectiveFunction(self, p_basegoal)

    # there is nothing to update in a point-to-point task
    def updateCosts(self, data) -> None:
        pass

    def updateGoalInModels(self, goal) -> None:
        T_w_lgoal, T_w_rgoal, p_basegoal = goal
        super().updateGoalInModels((T_w_lgoal, T_w_rgoal))
        BaseIKOCP.updateGoalInModels(self, p_basegoal)

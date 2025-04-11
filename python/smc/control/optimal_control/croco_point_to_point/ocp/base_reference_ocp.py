from smc.control.optimal_control.abstract_croco_ocp import CrocoOCP
from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface

import numpy as np
from pinocchio import SE3
import crocoddyl
from argparse import Namespace


class BaseIKOCP(CrocoOCP):
    def __init__(
        self,
        args: Namespace,
        robot: MobileBaseInterface,
        x0: np.ndarray,
        p_basegoal: SE3,
    ):
        super().__init__(args, robot, x0, p_basegoal)

    def constructTaskCostsValues(self):
        self.base_translation_cost_values = np.linspace(
            self.args.base_translation_cost,
            self.args.base_translation_cost
            * self.args.linearly_increasing_task_cost_factor,
            self.args.n_knots + 1,
        )

    def constructTaskObjectiveFunction(self, goal) -> None:
        p_basegoal = goal
        for i in range(self.args.n_knots):
            baseTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                self.state, self.robot.base_frame_id, p_basegoal, self.state.nv
            )
            baseTrackingCost = crocoddyl.CostModelResidual(
                self.state, baseTranslationResidual
            )
            self.runningCostModels[i].addCost(
                "base_translation" + str(i),
                baseTrackingCost,
                self.base_translation_cost_values[i],
            )
        self.terminalCostModel.addCost(
            "base_translation" + str(self.args.n_knots),
            baseTrackingCost,
            self.base_translation_cost_values[-1],
        )

    # there is nothing to update in a point-to-point task
    def updateCosts(self, data):
        pass

    def updateGoalInModels(self, goal) -> None:
        p_basegoal = goal
        for i, runningModel in enumerate(self.solver.problem.runningModels):
            runningModel.differential.costs.costs[
                "base_translation" + str(i)
            ].cost.residual.reference = p_basegoal
        self.solver.problem.terminalModel.differential.costs.costs[
            "base_translation" + str(self.args.n_knots)
        ].cost.residual.reference = p_basegoal

from smc.control.optimal_control.croco_point_to_point.ocp.base_and_dual_arm_reference_ocp import (
    BaseAndDualArmIKOCP,
)
from smc.robots.interfaces.whole_body_dual_arm_interface import (
    DualArmWholeBodyInterface,
)

from argparse import Namespace


class BaseAndDualArmEEPathFollowingOCP(BaseAndDualArmIKOCP):
    def __init__(self, args: Namespace, robot: DualArmWholeBodyInterface, x0):
        goal = robot.T_w_l, robot.T_w_r, robot.T_w_b.translation
        super().__init__(args, robot, x0, goal)

    def updateCosts(self, data):
        path_base = data[0]
        pathSE3_l = data[1]
        pathSE3_r = data[2]
        for i, runningModel in enumerate(self.solver.problem.runningModels):
            runningModel.differential.costs.costs[
                "base_translation" + str(i)
            ].cost.residual.reference = path_base[i]
            runningModel.differential.costs.costs[
                "l_ee_pose" + str(i)
            ].cost.residual.reference = pathSE3_l[i]
            runningModel.differential.costs.costs[
                "r_ee_pose" + str(i)
            ].cost.residual.reference = pathSE3_r[i]

        # idk if that's necessary
        self.solver.problem.terminalModel.differential.costs.costs[
            "base_translation" + str(self.args.n_knots)
        ].cost.residual.reference = path_base[-1]
        self.solver.problem.terminalModel.differential.costs.costs[
            "l_ee_pose" + str(self.args.n_knots)
        ].cost.residual.reference = pathSE3_l[-1]
        self.solver.problem.terminalModel.differential.costs.costs[
            "r_ee_pose" + str(self.args.n_knots)
        ].cost.residual.reference = pathSE3_r[-1]

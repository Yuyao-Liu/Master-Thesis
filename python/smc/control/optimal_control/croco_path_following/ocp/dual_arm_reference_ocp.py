from smc.control.optimal_control.croco_point_to_point.ocp.dual_arm_reference_ocp import (
    DualArmIKOCP,
)
from smc.robots.interfaces.dual_arm_interface import DualArmInterface

from argparse import Namespace


class DualArmEEPathFollowingOCP(DualArmIKOCP):
    def __init__(self, args: Namespace, robot: DualArmInterface, x0):
        goal = (robot.T_w_l, robot.T_w_r)
        super().__init__(args, robot, x0, goal)

    def updateCosts(self, data):
        pathSE3_l = data[0]
        pathSE3_r = data[1]
        for i, runningModel in enumerate(self.solver.problem.runningModels):
            runningModel.differential.costs.costs[
                "l_ee_pose" + str(i)
            ].cost.residual.reference = pathSE3_l[i]
            runningModel.differential.costs.costs[
                "r_ee_pose" + str(i)
            ].cost.residual.reference = pathSE3_r[i]

        self.solver.problem.terminalModel.differential.costs.costs[
            "l_ee_pose" + str(self.args.n_knots)
        ].cost.residual.reference = pathSE3_l[-1]
        self.solver.problem.terminalModel.differential.costs.costs[
            "r_ee_pose" + str(self.args.n_knots)
        ].cost.residual.reference = pathSE3_r[-1]

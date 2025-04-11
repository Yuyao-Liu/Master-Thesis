from smc.control.optimal_control.croco_point_to_point.ocp.base_reference_ocp import (
    BaseIKOCP,
)
from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface

from argparse import Namespace


class BasePathFollowingOCP(BaseIKOCP):
    """
    createBaseAndEEPathFollowingOCP
    -------------------------------
    creates a path following problem.
    it is instantiated to just to stay at the current position.
    NOTE: the path MUST be time indexed with the SAME time used between the knots
    """

    def __init__(self, args: Namespace, robot: MobileBaseInterface, x0):
        goal = robot.T_w_b.translation.copy()
        super().__init__(args, robot, x0, goal)

    def updateCosts(self, data):
        path_base = data
        for i, runningModel in enumerate(self.solver.problem.runningModels):
            runningModel.differential.costs.costs[
                "base_translation" + str(i)
            ].cost.residual.reference = path_base[i]

        # idk if that's necessary
        self.solver.problem.terminalModel.differential.costs.costs[
            "base_translation" + str(self.args.n_knots)
        ].cost.residual.reference = path_base[-1]

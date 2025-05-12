from smc.robots.abstract_robotmanager import AbstractRobotManager

import numpy as np
import crocoddyl
from argparse import Namespace
import importlib.util
import abc
from typing import Any

from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface

if importlib.util.find_spec("mim_solvers"):
    try:
        import mim_solvers
    except ModuleNotFoundError:
        print(
            "mim_solvers installation is broken, rebuild and reinstall it if you want it"
        )


class CrocoOCP(abc.ABC):
    """
    CrocoOCP
    ----------
    General info:
        - torque inputs are assumed - we already solve for state, which includes velocity commands, and we send that if the robot accepts vel cmds
    State model : StateMultibody
        I'm not even sure what else is there to choose lol.
    Actuation model : ActuatioModelFull
        We do underactuation via a constraint at the moment. This is solely
        because coding up a proper underactuated model is annoying,
        and dealing with this is a TODO for later. Having that said,
        input constraints are necessary either way.
            # TODO: consider ActuationModelFloatingBaseTpl for heron
            # TODO: create a different actuation model (or whatever)
            # for the mobile base - basically just remove the y movement in the base
            # and update the corresponding derivates to 0
            # there's python examples for this, ex. acrobot.
            # you might want to implement the entire action model too idk what's really necessary here
    Action model : DifferentialActionModelFreeFwdDynamics
        We need to create an action model for running and terminal knots. The
        forward dynamics (computed using ABA) are implemented
        inside DifferentialActionModelFreeFwdDynamics.
    """

    def __init__(
        self, args: Namespace, robot: AbstractRobotManager, x0: np.ndarray, goal
    ):
        # TODO: declare attributes here
        self.args = args
        self.solver_name = args.solver
        self.robot = robot
        self.x0 = x0
        self.constructCrocoCostModels()
        self.constructRegulationCosts()
        self.constructStateLimits()
        self.constructConstraints()
        self.constructTaskCostsValues()
        self.constructTaskObjectiveFunction(goal)
        self.constructRunningModels()
        self.problem = crocoddyl.ShootingProblem(
            self.x0, self.runningModels, self.terminalModel
        )
        self.createSolver()

    def constructCrocoCostModels(self):
        self.state = crocoddyl.StateMultibody(self.robot.model)
        # NOTE: atm we just set effort level in that direction to 0
        self.actuation = crocoddyl.ActuationModelFull(self.state)

        # we will be summing 4 different costs
        # first 3 are for tracking, state and control regulation
        self.runningCostModels = []
        for _ in range(self.args.n_knots):
            self.runningCostModels.append(crocoddyl.CostModelSum(self.state))
        self.terminalCostModel = crocoddyl.CostModelSum(self.state)

    def constructRegulationCosts(self):
        # cost 1) u residual (actuator cost)
        self.uResidual = crocoddyl.ResidualModelControl(self.state, self.state.nv)
        self.uRegCost = crocoddyl.CostModelResidual(self.state, self.uResidual)
        # cost 2) x residual (overall amount of movement)
        self.xResidual = crocoddyl.ResidualModelState(
            self.state, self.x0, self.state.nv
        )
        self.xRegCost = crocoddyl.CostModelResidual(self.state, self.xResidual)

        # put these costs into the running costs
        for i in range(self.args.n_knots):
            self.runningCostModels[i].addCost(
                "xReg", self.xRegCost, self.args.u_reg_cost
            )
            self.runningCostModels[i].addCost(
                "uReg", self.uRegCost, self.args.x_reg_cost
            )
        # and add the terminal cost, which is the distance to the goal
        # NOTE: shouldn't there be a final velocity = 0 here?
        # --> no if you're not stopping at the last knot!
        self.terminalCostModel.addCost("uReg", self.uRegCost, 1e3)

    def constructStateLimits(self) -> None:
        """
        constructStateConstraints
        -------------------------
        """
        # the 4th cost is for defining bounds via costs
        # NOTE: could have gotten the same info from state.lb and state.ub.
        # the first state is unlimited there idk what that means really,
        # but the arm's base isn't doing a full rotation anyway, let alone 2 or more
        self.xlb = np.concatenate(
            [self.robot.model.lowerPositionLimit, -1 * self.robot._max_v]
        )
        self.xub = np.concatenate(
            [self.robot.model.upperPositionLimit, self.robot._max_v]
        )

        # NOTE: in an ideal universe this is handled elsewhere
        # we have no limits on the mobile base.
        # the mobile base is a planar joint.
        # since it is represented as [x,y,cos(theta),sin(theta)], there is no point
        # to limiting cos(theta),sin(theta) even if we wanted limits,
        # because we would then limit theta, or limit ct and st jointly.
        # in any event, xlb and xub are 1 number too long --
        # the residual has to be [x,y,theta] because it is in the tangent space -
        # the difference between two points on a manifold in pinocchio is defined
        # as the velocity which if parallel transported for 1 unit of "time"
        # from one to point to the other.
        # point activation input and the residual need to be of the same length obviously,
        # and this should be 2 * model.nv the way things are defined here.

        if issubclass(self.robot.__class__, MobileBaseInterface):
            self.xlb = self.xlb[1:]
            self.xub = self.xub[1:]

    def constructConstraints(self) -> None:
        if self.solver_name == "boxfddp":
            self.stateConstraintsViaBarriersInObjectiveFunction()
        if self.solver_name == "csqp":
            self.boxInputConstraintsAsActualConstraints()

    # NOTE: used by BoxFDDP
    def stateConstraintsViaBarriersInObjectiveFunction(self) -> None:
        bounds = crocoddyl.ActivationBounds(self.xlb, self.xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(
            self.state, self.x0, self.state.nv
        )
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

        limitCost = crocoddyl.CostModelResidual(
            self.state, xLimitActivation, xLimitResidual
        )
        for i in range(self.args.n_knots):
            self.runningCostModels[i].addCost("limitCost", limitCost, 1e3)
        self.terminalCostModel.addCost("limitCost", limitCost, 1e3)

    # NOTE: used by csqp
    def boxInputConstraintsAsActualConstraints(self) -> None:
        # ConstraintModelManager just stores constraints
        self.constraints = crocoddyl.ConstraintModelManager(
            self.state, self.robot.model.nv
        )
        u_constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            self.uResidual,
            -1 * self.robot.model.effortLimit * 0.1,
            self.robot.model.effortLimit * 0.1,
        )
        self.constraints.addConstraint("u_box_constraint", u_constraint)
        x_constraint = crocoddyl.ConstraintModelResidual(
            self.state, self.xResidual, self.xlb, self.xub
        )
        self.constraints.addConstraint("x_box_constraint", x_constraint)

    def constructRunningModels(self) -> None:
        self.runningModels = []
        if self.solver_name == "boxfddp":
            self.constructRunningModelsWithoutExplicitConstraints()
        if self.solver_name == "csqp":
            self.constructRunningModelsWithExplicitConstraints()

    # NOTE: has input constraints, but reads them from model.effortlimit
    # or whatever else
    def constructRunningModelsWithoutExplicitConstraints(self) -> None:
        for i in range(self.args.n_knots):
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    self.state, self.actuation, self.runningCostModels[i]
                ),
                self.args.ocp_dt,
            )
            runningModel.u_lb = -1 * self.robot.model.effortLimit * 0.1
            runningModel.u_ub = self.robot.model.effortLimit * 0.1
            self.runningModels.append(runningModel)
        self.terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                self.state, self.actuation, self.terminalCostModel
            ),
            0.0,
        )
        self.terminalModel.u_lb = -1 * self.robot.model.effortLimit * 0.1
        self.terminalModel.u_ub = self.robot.model.effortLimit * 0.1

    def constructRunningModelsWithExplicitConstraints(self) -> None:
        for i in range(self.args.n_knots):
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    self.state,
                    self.actuation,
                    self.runningCostModels[i],
                    self.constraints,
                ),
                self.args.ocp_dt,
            )
            self.runningModels.append(runningModel)
        self.terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                self.state, self.actuation, self.terminalCostModel, self.constraints
            ),
            0.0,
        )

    @abc.abstractmethod
    def constructTaskCostsValues(self) -> None: ...

    @abc.abstractmethod
    def constructTaskObjectiveFunction(self, goal) -> None: ...

    def getSolver(self) -> Any:
        return self.solver

    def createSolver(self) -> None:
        if self.solver_name == "boxfddp":
            self.createCrocoSolver()
        if self.solver_name == "csqp":
            self.createMimSolver()

    # NOTE: used by boxfddp
    def createCrocoSolver(self) -> None:
        # just for reference can try
        # solver = crocoddyl.SolverIpopt(problem)
        # TODO: select other solvers from arguments
        self.solver = crocoddyl.SolverBoxFDDP(self.problem)

    #        if self.args.debug_prints:
    #            self.solver.setCallbacks(
    #                [crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()]
    #            )

    # NOTE: used by csqp
    def createMimSolver(self) -> None:
        # TODO try out the following solvers from mim_solvers:
        #   - csqp
        #   - stagewise qp
        # and the solver
        # both of these have generic inequalities you can put in.
        # and both are basically QP versions of iLQR if i'm not wrong
        # (i have no idea tho)
        # solver = mim_solvers.SolverSQP(problem)
        # TODO: select other solvers from arguments
        self.solver = mim_solvers.SolverCSQP(self.problem)

    #        if self.args.debug_prints:
    #            self.solver.setCallbacks(
    #                [mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()]
    #            )

    # this shouldn't really depend on x0 but i can't be bothered
    def solveInitialOCP(self, x0: np.ndarray):
        # run solve
        # NOTE: there are some possible parameters here that I'm not using
        xs = [x0] * (self.solver.problem.T + 1)
        us = self.solver.problem.quasiStatic([x0] * self.solver.problem.T)

        # start = time.time()
        # self.solver.solve(xs, us, 500, False, 1e-9)
        self.solver.solve(xs, us, self.args.max_solver_iter)
        # end = time.time()
        # print("solved in:", end - start, "seconds")

    def getSolvedReference(self) -> dict[str, Any]:
        # solver.solve()
        # get reference (state trajectory)
        # we aren't using controls because we only have velocity inputs
        xs = np.array(self.solver.xs)
        qs = xs[:, : self.robot.model.nq]
        vs = xs[:, self.robot.model.nq :]
        reference = {"qs": qs, "vs": vs, "dt": self.args.ocp_dt}
        return reference

    # NOTE: this is ugly, but idk how to deal with the fact that i don't know
    # which kind of arguments this function needs
    def updateCosts(self, data):
        raise NotImplementedError(
            "if you want to warmstart and resolve, you need \
            to specify how do you update the cost function (could be nothing) \
            in between resolving"
        )

    def warmstartAndReSolve(self, x0: np.ndarray, data=None) -> None:
        self.solver.problem.x0 = x0
        xs_init = list(self.solver.xs[1:]) + [self.solver.xs[-1]]
        xs_init[0] = x0
        us_init = list(self.solver.us[1:]) + [self.solver.us[-1]]
        self.updateCosts(data)
        self.solver.solve(xs_init, us_init, self.args.max_solver_iter)

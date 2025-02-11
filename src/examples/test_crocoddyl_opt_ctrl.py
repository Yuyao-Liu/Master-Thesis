import os
import signal
import sys
import time

from ur_simple_control.util.get_model import get_model
import numpy as np
import pinocchio
from pinocchio.visualize import MeshcatVisualizer

import crocoddyl

model, collision_model, visual_model, data = get_model()
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)
#q0 = np.zeros(model.nq)
pinocchio.seed(int(time.time()))
q0 = pinocchio.randomConfiguration(model)
x0 = np.concatenate([q0, pinocchio.utils.zero(model.nv)])


# Create a cost model per the running and terminal action model.
nu = state.nv
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    model.getFrameId("tool0"),
    pinocchio.SE3(np.eye(3), np.array([0.6, 0.2, 0.5])),
    nu,
)
uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
xResidual = crocoddyl.ResidualModelState(state, x0, nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)


# Then let's added the running and terminal cost functions
# NOTE: there's a 10e4 diff between param sizes, is that ok?
runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e-3)
# shouldn't this be a constraint tho?
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(
        state, actuation, runningCostModel
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(
        state, actuation, terminalCostModel
    ),
    0.0,
)

# For this optimal control problem, we define 100 knots (or running action
# models) plus a terminal knot
T = 100
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)


# Creating the solver for this OC problem, defining a logger
#solver = crocoddyl.SolverIntro(problem)
#solver = crocoddyl.SolverFDDP(problem)
solver = crocoddyl.SolverDDP(problem)
#solver = crocoddyl.SolverKKT(problem)
solver.setCallbacks(
    [
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackLogger(),
    ]
)

solver.solve()
print(
    "Finally reached = ",
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        model.getFrameId("tool0")
    ].translation.T,
)


log = solver.getCallbacks()[1]
crocoddyl.plotOCSolution(
    solver.xs,
    [d.differential.multibody.joint.tau for d in solver.problem.runningDatas],
    figIndex=1,
    show=False,
)
crocoddyl.plotConvergence(
    log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
)

# viz

for geom in visual_model.geometryObjects:
    if "hand" in geom.name:
        s = geom.meshScale
        # this looks exactly correct lmao
        s *= 0.001
        geom.meshScale = s
for geom in collision_model.geometryObjects:
    if "hand" in geom.name:
        s = geom.meshScale
        # this looks exactly correct lmao
        s *= 0.001
        geom.meshScale = s
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()
xs = solver.xs.tolist()
for i in range(10):
    for x in xs:
        viz.display(x[0:model.nq])
        time.sleep(0.05)
#display = crocoddyl.MeshcatDisplay(model)#, collision_model, visual_model)
#display.rate = -1
#display.freq = 1
#while True:
#    display.displayFromSolver(solver)
#    time.sleep(1.0)

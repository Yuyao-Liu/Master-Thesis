import numpy as np
import pinocchio as pin
import crocoddyl
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
import argparse
from ur_simple_control.basics.basics import followKinematicJointTrajP
from ur_simple_control.visualize.visualize import plotFromDict
import example_robot_data
import time
import importlib.util
if importlib.util.find_spec('mim_solvers'):
    import mim_solvers

def createCrocoIKOCP(args, robot : RobotManager, x0, goal : pin.SE3):
    if robot.robot_name == "yumi":
        goal_l, goal_r = goal
    # create torque bounds which correspond to percentage
    # of maximum allowed acceleration 
    # NOTE: idk if this makes any sense, but let's go for it
    #print(robot.model.effortLimit)
    #exit()
    #robot.model.effortLimit = robot.model.effortLimit * (args.acceleration / robot.MAX_ACCELERATION)
    #robot.model.effortLimit = robot.model.effortLimit * 0.5
    #robot.data = robot.model.createData()
    # TODO: make it underactuated in mobile base's y-direction
    state = crocoddyl.StateMultibody(robot.model)
    # command input IS torque 
    # TODO: consider ActuationModelFloatingBaseTpl for heron
    # TODO: create a different actuation model (or whatever)
    # for the mobile base - basically just remove the y movement in the base
    # and update the corresponding derivates to 0
    # there's python examples for this, ex. acrobot.
    # you might want to implement the entire action model too idk what's really necessary here
    actuation = crocoddyl.ActuationModelFull(state)

    # we will be summing 4 different costs
    # first 3 are for tracking, state and control regulation
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    # cost 1) u residual (actuator cost)
    uResidual = crocoddyl.ResidualModelControl(state, state.nv)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # cost 2) x residual (overall amount of movement)
    xResidual = crocoddyl.ResidualModelState(state, x0, state.nv)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # cost 3) distance to goal -> SE(3) error
    # TODO: make this follow a path.
    # to do so, you need to implement a residualmodel for that in crocoddyl.
    # there's an example of exending crocoddyl in colmpc repo
    # (look at that to see how to compile, make python bindings etc)
    if robot.robot_name != "yumi":
        framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                state,
                # TODO: check if this is the frame we actually want (ee tip)
                # the id is an integer and that's what api wants
                robot.model.getFrameId("tool0"),
                goal.copy(),
                state.nv)
        goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
        # cost 4) final ee velocity is 0 (can't directly formulate joint velocity,
        #         because that's a part of the state, and we don't know final joint positions)
        frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                state,
                robot.model.getFrameId("tool0"),
                pin.Motion(np.zeros(6)),
                pin.ReferenceFrame.WORLD)
        frameVelocityCost = crocoddyl.CostModelResidual(state, frameVelocityResidual)
        runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
        terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
        terminalCostModel.addCost("velFinal", frameVelocityCost, 1e3)
    else:
        framePlacementResidual_l = crocoddyl.ResidualModelFramePlacement(
                state,
                # TODO: check if this is the frame we actually want (ee tip)
                # the id is an integer and that's what api wants
                robot.model.getFrameId("robl_joint_7"),
                goal_l.copy(),
                state.nv)
        framePlacementResidual_r = crocoddyl.ResidualModelFramePlacement(
                state,
                # TODO: check if this is the frame we actually want (ee tip)
                # the id is an integer and that's what api wants
                robot.model.getFrameId("robr_joint_7"),
                goal_r.copy(),
                state.nv)
        goalTrackingCost_l = crocoddyl.CostModelResidual(state, framePlacementResidual_l)
        goalTrackingCost_r = crocoddyl.CostModelResidual(state, framePlacementResidual_r)
        frameVelocityResidual_l = crocoddyl.ResidualModelFrameVelocity(
                state,
                robot.model.getFrameId("robl_joint_7"),
                pin.Motion(np.zeros(6)),
                pin.ReferenceFrame.WORLD)
        frameVelocityResidual_r = crocoddyl.ResidualModelFrameVelocity(
                state,
                robot.model.getFrameId("robr_joint_7"),
                pin.Motion(np.zeros(6)),
                pin.ReferenceFrame.WORLD)
        frameVelocityCost_l = crocoddyl.CostModelResidual(state, frameVelocityResidual_l)
        frameVelocityCost_r = crocoddyl.CostModelResidual(state, frameVelocityResidual_r)
        runningCostModel.addCost("gripperPose_l", goalTrackingCost_l, 1e2)
        runningCostModel.addCost("gripperPose_r", goalTrackingCost_r, 1e2)
        terminalCostModel.addCost("gripperPose_l", goalTrackingCost_l, 1e2)
        terminalCostModel.addCost("gripperPose_r", goalTrackingCost_r, 1e2)
        terminalCostModel.addCost("velFinal_l", frameVelocityCost_l, 1e3)
        terminalCostModel.addCost("velFinal_r", frameVelocityCost_r, 1e3)

    # put these costs into the running costs
    runningCostModel.addCost("xReg", xRegCost, 1e-3)
    runningCostModel.addCost("uReg", uRegCost, 1e-3)
    # and add the terminal cost, which is the distance to the goal
    # NOTE: shouldn't there be a final velocity = 0 here?
    terminalCostModel.addCost("uReg", uRegCost, 1e3)

    ######################################################################
    #  state constraints  #
    #################################################
    # - added to costs via barrier functions for fddp
    # - added as actual constraints via crocoddyl.constraintmanager for csqp
    ###########################################################################
    

    # the 4th cost is for defining bounds via costs
    # NOTE: could have gotten the same info from state.lb and state.ub.
    # the first state is unlimited there idk what that means really,
    # but the arm's base isn't doing a full rotation anyway, let alone 2 or more
    xlb = np.concatenate([
        robot.model.lowerPositionLimit,
        -1 * robot.model.velocityLimit])
    xub = np.concatenate([
        robot.model.upperPositionLimit,
        robot.model.velocityLimit])
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


    if robot.robot_name == "heron" or robot.robot_name == "heronros" or robot.robot_name == "mirros" or robot.robot_name == "yumi":
        xlb = xlb[1:]
        xub = xub[1:]

    # TODO: make these constraints-turned-to-objectives for end-effector following
    # the handlebar position
    if args.solver == "boxfddp":
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(state, x0, state.nv)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

        limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)
        runningCostModel.addCost("limitCost", limitCost, 1e3)
        terminalCostModel.addCost("limitCost", limitCost, 1e3)

    # TODO: try using crocoddyl.ConstraintModelResidual
    # instead to create actual box constraints (inequality constraints)
    # TODO: same goes for control input
    # NOTE: i'm not sure whether any solver uses these tho lel 
    # --> you only do that for mim_solvers' csqp!

    if args.solver == "csqp":
        # this just store all the constraints
        constraints = crocoddyl.ConstraintModelManager(state, robot.model.nv)
        u_constraint = crocoddyl.ConstraintModelResidual(
                state,
                uResidual, 
                -1 * robot.model.effortLimit * 0.1,
                robot.model.effortLimit  * 0.1)
        constraints.addConstraint("u_box_constraint", u_constraint)
        x_constraint = crocoddyl.ConstraintModelResidual(
                state,
                xResidual, 
                xlb,
                xub)
        constraints.addConstraint("x_box_constraint", x_constraint)


    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using ABA) are implemented
    # inside DifferentialActionModelFreeFwdDynamics.
    if args.solver == "boxfddp":
        runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, runningCostModel
            ),
            args.ocp_dt,
        )
        runningModel.u_lb = -1 * robot.model.effortLimit * 0.1
        runningModel.u_ub = robot.model.effortLimit  * 0.1
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, terminalCostModel
            ),
            0.0,
        )
        terminalModel.u_lb = -1 * robot.model.effortLimit * 0.1 
        terminalModel.u_ub = robot.model.effortLimit  * 0.1
    if args.solver == "csqp":
        runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, runningCostModel, constraints
            ),
            args.ocp_dt,
        )
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, terminalCostModel, constraints
            ),
            0.0,
        )


    # now we define the problem
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * args.n_knots, terminalModel)
    return problem 

# this shouldn't really depend on x0 but i can't be bothered
def solveCrocoOCP(args, robot, problem, x0):
    # and the solver
    # TODO try out the following solvers from mim_solvers:
    #   - csqp
    #   - stagewise qp
    # both of these have generic inequalities you can put in.
    # and both are basically QP versions of iLQR if i'm not wrong
    # (i have no idea tho)
    if args.solver == "boxfddp":
        solver = crocoddyl.SolverBoxFDDP(problem)
    if args.solver == "csqp":
        solver = mim_solvers.SolverCSQP(problem)
    #solver = mim_solvers.SolverSQP(problem)
    #solver = crocoddyl.SolverIpopt(problem)
    # TODO: remove / place elsewhere once it works (make it an argument)
    if args.solver == "boxfddp":
        solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
    if args.solver == "csqp":
        solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
    # run solve
    # NOTE: there are some possible parameters here that I'm not using
    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)

    start = time.time()
    solver.solve(xs, us, 500, False, 1e-9)
    end = time.time()
    print("solved in:", end - start, "seconds")

    #solver.solve()
    # get reference (state trajectory)
    # we aren't using controls because we only have velocity inputs
    xs = np.array(solver.xs)
    qs = xs[:, :robot.model.nq]
    vs = xs[:, robot.model.nq:]
    reference = {'qs' : qs, 'vs' : vs, 'dt' : args.ocp_dt}
    return reference, solver



def createCrocoEEPathFollowingOCP(args, robot : RobotManager, x0):
    """
    createCrocoEEPathFollowingOCP
    -------------------------------
    creates a path following problem with a single end-effector reference.
    it is instantiated to just to stay at the current position.
    NOTE: the path MUST be time indexed with the SAME time used between the knots
    """
    T_w_e = robot.getT_w_e()
    path = [T_w_e] * args.n_knots
    # underactuation is done by setting max-torque to 0 in the robot model,
    # and since we torques are constrained we're good
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)

    # we will be summing 4 different costs
    # first 3 are for tracking, state and control regulation
    terminalCostModel = crocoddyl.CostModelSum(state)
    # cost 1) u residual (actuator cost)
    uResidual = crocoddyl.ResidualModelControl(state, state.nv)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # cost 2) x residual (overall amount of movement)
    xResidual = crocoddyl.ResidualModelState(state, x0, state.nv)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # cost 4) final ee velocity is 0 (can't directly formulate joint velocity,
    #         because that's a part of the state, and we don't know final joint positions)
    #frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
    #        state,
    #        robot.model.getFrameId("tool0"),
    #        pin.Motion(np.zeros(6)),
    #        pin.ReferenceFrame.WORLD)
    #frameVelocityCost = crocoddyl.CostModelResidual(state, frameVelocityResidual)

    # put these costs into the running costs
    # we put this one in later
    # and add the terminal cost, which is the distance to the goal
    # NOTE: shouldn't there be a final velocity = 0 here?
    terminalCostModel.addCost("uReg", uRegCost, 1e3)
    #terminalCostModel.addCost("velFinal", frameVelocityCost, 1e3)

    ######################################################################
    #  state constraints  #
    #################################################
    # - added to costs via barrier functions for fddp
    # - added as actual constraints via crocoddyl.constraintmanager for csqp
    ###########################################################################
    

    # the 4th cost is for defining bounds via costs
    # NOTE: could have gotten the same info from state.lb and state.ub.
    # the first state is unlimited there idk what that means really,
    # but the arm's base isn't doing a full rotation anyway, let alone 2 or more
    xlb = np.concatenate([
        robot.model.lowerPositionLimit,
        -1 * robot.model.velocityLimit])
    xub = np.concatenate([
        robot.model.upperPositionLimit,
        robot.model.velocityLimit])

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
    if robot.robot_name == "heron" or robot.robot_name == "heronros" or robot.robot_name == "mirros":
        xlb = xlb[1:]
        xub = xub[1:]

    # TODO: make these constraints-turned-to-objectives for end-effector following
    # the handlebar position
    if args.solver == "boxfddp":
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(state, x0, state.nv)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

        limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)
        terminalCostModel.addCost("limitCost", limitCost, 1e3)

    # csqp actually allows us to put in hard constraints so we do that
    if args.solver == "csqp":
        # this just store all the constraints
        constraints = crocoddyl.ConstraintModelManager(state, robot.model.nv)
        u_constraint = crocoddyl.ConstraintModelResidual(
                state,
                uResidual, 
                -1 * robot.model.effortLimit * 0.1,
                robot.model.effortLimit  * 0.1)
        constraints.addConstraint("u_box_constraint", u_constraint)
        x_constraint = crocoddyl.ConstraintModelResidual(
                state,
                xResidual, 
                xlb,
                xub)
        constraints.addConstraint("x_box_constraint", x_constraint)


    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using ABA) are implemented
    # inside DifferentialActionModelFreeFwdDynamics.

    runningModels = []
    for i in range(args.n_knots):
        runningCostModel = crocoddyl.CostModelSum(state)
        runningCostModel.addCost("xReg", xRegCost, 1e-3)
        runningCostModel.addCost("uReg", uRegCost, 1e-3)
        if args.solver == "boxfddp":
            runningCostModel.addCost("limitCost", limitCost, 1e3)
        framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                state,
                robot.model.getFrameId("tool0"),
                path[i], # this better be done with the same time steps as the knots
                         # NOTE: this should be done outside of this function to clarity
                state.nv)
        goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
        runningCostModel.addCost("gripperPose" + str(i), goalTrackingCost, 1e2)
        #runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)

        if args.solver == "boxfddp":
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    state, actuation, runningCostModel
                ),
                args.ocp_dt,
            )
            runningModel.u_lb = -1 * robot.model.effortLimit * 0.1
            runningModel.u_ub = robot.model.effortLimit  * 0.1
        if args.solver == "csqp":
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    state, actuation, runningCostModel, constraints
                ),
                args.ocp_dt,
            )
        runningModels.append(runningModel)

    # terminal model
    if args.solver == "boxfddp":
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, terminalCostModel
            ),
            0.0,
        )
        terminalModel.u_lb = -1 * robot.model.effortLimit * 0.1 
        terminalModel.u_ub = robot.model.effortLimit  * 0.1
    if args.solver == "csqp":
            terminalModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    state, actuation, terminalCostModel, constraints
                ),
                0.0,
            )

    terminalCostModel.addCost("gripperPose" + str(args.n_knots), goalTrackingCost, 1e2)
    #terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e2)

    # now we define the problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    return problem 

def createBaseAndEEPathFollowingOCP(args, robot : RobotManager, x0):
    """
    createBaseAndEEPathFollowingOCP
    -------------------------------
    creates a path following problem.
    it is instantiated to just to stay at the current position.
    NOTE: the path MUST be time indexed with the SAME time used between the knots
    """
    if robot.robot_name != "yumi":
        T_w_e = robot.getT_w_e()
        path_ee = [T_w_e] * args.n_knots
    else:
        T_w_e_left, T_w_e_right = robot.getT_w_e()
        path_ee = [T_w_e_left] * args.n_knots
    # TODO: have a different reference for each arm
    path_base = [np.append(x0[:2], 0.0)] * args.n_knots
    # underactuation is done by setting max-torque to 0 in the robot model,
    # and since we torques are constrained we're good
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)

    # we will be summing 6 different costs
    # first 3 are for tracking, state and control regulation,
    # the others depend on the path and will be defined later
    terminalCostModel = crocoddyl.CostModelSum(state)
    # cost 1) u residual (actuator cost)
    uResidual = crocoddyl.ResidualModelControl(state, state.nv)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # cost 2) x residual (overall amount of movement)
    xResidual = crocoddyl.ResidualModelState(state, x0, state.nv)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)

    # put these costs into the running costs
    # we put this one in later
    # and add the terminal cost, which is the distance to the goal
    terminalCostModel.addCost("uReg", uRegCost, 1e3)

    ######################################################################
    #  state constraints  #
    #################################################
    # - added to costs via barrier functions for fddp (4th cost function)
    # - added as actual constraints via crocoddyl.constraintmanager for csqp
    ###########################################################################
    xlb = np.concatenate([
        robot.model.lowerPositionLimit,
        -1 * robot.model.velocityLimit])
    xub = np.concatenate([
        robot.model.upperPositionLimit,
        robot.model.velocityLimit])
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


    if robot.robot_name == "heron" or robot.robot_name == "heronros" or robot.robot_name == "mirros" or robot.robot_name == "yumi":
        xlb = xlb[1:]
        xub = xub[1:]

    # TODO: make these constraints-turned-to-objectives for end-effector following
    # the handlebar position
    if args.solver == "boxfddp":
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(state, x0, state.nv)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

        limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)
        terminalCostModel.addCost("limitCost", limitCost, 1e3)


    # csqp actually allows us to put in hard constraints so we do that
    if args.solver == "csqp":
        # this just store all the constraints
        constraints = crocoddyl.ConstraintModelManager(state, robot.model.nv)
        u_constraint = crocoddyl.ConstraintModelResidual(
                state,
                uResidual, 
                -1 * robot.model.effortLimit * 0.1,
                robot.model.effortLimit  * 0.1)
        constraints.addConstraint("u_box_constraint", u_constraint)
        x_constraint = crocoddyl.ConstraintModelResidual(
                state,
                xResidual, 
                xlb,
                xub)
        constraints.addConstraint("x_box_constraint", x_constraint)

    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using ABA) are implemented
    # inside DifferentialActionModelFreeFwdDynamics.
    runningModels = []
    for i in range(args.n_knots):
        runningCostModel = crocoddyl.CostModelSum(state)
        runningCostModel.addCost("xReg", xRegCost, 1e-3)
        runningCostModel.addCost("uReg", uRegCost, 1e-3)
        if args.solver == "boxfddp":
            runningCostModel.addCost("limitCost", limitCost, 1e3)

        ##########################
        #  single arm reference  #
        ##########################
        if robot.robot_name != "yumi":
            eePoseResidual = crocoddyl.ResidualModelFramePlacement(
                    state,
                    robot.model.getFrameId("tool0"),
                    path_ee[i], # this better be done with the same time steps as the knots
                             # NOTE: this should be done outside of this function to clarity
                    state.nv)
            eeTrackingCost = crocoddyl.CostModelResidual(state, eePoseResidual)
            runningCostModel.addCost("ee_pose" + str(i), eeTrackingCost, args.ee_pose_cost)
        #########################
        #  dual arm references  #
        #########################
        else:
            l_eePoseResidual = crocoddyl.ResidualModelFramePlacement(
                    state,
                    robot.model.getFrameId("robl_joint_7"),
                    # MASSIVE TODO: actually put in reference for left arm
                    path_ee[i],
                    state.nv)
            l_eeTrackingCost = crocoddyl.CostModelResidual(state, l_eePoseResidual)
            runningCostModel.addCost("l_ee_pose" + str(i), l_eeTrackingCost, args.ee_pose_cost)
            r_eePoseResidual = crocoddyl.ResidualModelFramePlacement(
                    state,
                    robot.model.getFrameId("robr_joint_7"),
                    # MASSIVE TODO: actually put in reference for left arm
                    path_ee[i], 
                    state.nv)
            r_eeTrackingCost = crocoddyl.CostModelResidual(state, r_eePoseResidual)
            runningCostModel.addCost("r_ee_pose" + str(i), r_eeTrackingCost, args.ee_pose_cost)


        baseTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                state,
                robot.model.getFrameId("mobile_base"),
                path_base[i],
                state.nv)
        baseTrackingCost = crocoddyl.CostModelResidual(state, baseTranslationResidual)
        runningCostModel.addCost("base_translation" + str(i), baseTrackingCost, args.base_translation_cost)

        if args.solver == "boxfddp":
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    state, actuation, runningCostModel
                ),
                args.ocp_dt,
            )
            runningModel.u_lb = -1 * robot.model.effortLimit * 0.1
            runningModel.u_ub = robot.model.effortLimit  * 0.1
        if args.solver == "csqp":
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    state, actuation, runningCostModel, constraints
                ),
                args.ocp_dt,
            )
        runningModels.append(runningModel)

    # terminal model
    if args.solver == "boxfddp":
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, terminalCostModel
            ),
            0.0,
        )
        terminalModel.u_lb = -1 * robot.model.effortLimit * 0.1 
        terminalModel.u_ub = robot.model.effortLimit  * 0.1
    if args.solver == "csqp":
            terminalModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeInvDynamics(
                    state, actuation, terminalCostModel, constraints
                ),
                0.0,
            )

    if robot.robot_name != "yumi":
        terminalCostModel.addCost("ee_pose" + str(args.n_knots), eeTrackingCost, args.ee_pose_cost)
    else:
        terminalCostModel.addCost("l_ee_pose" + str(args.n_knots), l_eeTrackingCost, args.ee_pose_cost)
        terminalCostModel.addCost("r_ee_pose" + str(args.n_knots), r_eeTrackingCost, args.ee_pose_cost)
    terminalCostModel.addCost("base_translation" + str(args.n_knots), baseTrackingCost, args.base_translation_cost)

    # now we define the problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    return problem 

if __name__ == "__main__":
    parser = getMinimalArgParser()
    parser = get_OCP_args(parser)
    args = parser.parse_args()
    ex_robot = example_robot_data.load("ur5_gripper")
    robot = RobotManager(args)
    # TODO: remove once things work
    robot.max_qd = 3.14 
    print("velocity limits", robot.model.velocityLimit)
    robot.q = pin.randomConfiguration(robot.model)
    robot.q[0] = 0.1
    robot.q[1] = 0.1
    print(robot.q)
    goal = pin.SE3.Random()
    goal.translation = np.random.random(3) * 0.4
    reference, solver = CrocoIKOCP(args, robot, goal)
    # we only work within -pi - pi (or 0-2pi idk anymore)
    #reference['qs'] = reference['qs'] % (2*np.pi) - np.pi
    if args.solver == "boxfddp":
        log = solver.getCallbacks()[1]
        crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=True)
        crocoddyl.plotConvergence(log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2)
    followKinematicJointTrajP(args, robot, reference)
    reference.pop('dt')
    plotFromDict(reference, args.n_knots + 1, args)
    print("achieved result", robot.getT_w_e())
    display = crocoddyl.MeshcatDisplay(ex_robot)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)

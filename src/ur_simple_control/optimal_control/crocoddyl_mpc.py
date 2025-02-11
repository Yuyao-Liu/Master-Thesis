from ur_simple_control.managers import (
    getMinimalArgParser,
    ControlLoopManager,
    RobotManager,
    ProcessManager,
)
from ur_simple_control.optimal_control.crocoddyl_optimal_control import (
    createCrocoIKOCP,
    createCrocoEEPathFollowingOCP,
    createBaseAndEEPathFollowingOCP,
)
import pinocchio as pin
import crocoddyl
import numpy as np
from functools import partial
import types
import importlib.util

if importlib.util.find_spec("mim_solvers"):
    import mim_solvers

# TODO: put others here too
if importlib.util.find_spec("shapely"):
    from ur_simple_control.path_generation.planner import (
        path2D_timed,
        pathPointFromPathParam,
        path2D_to_SE3,
    )


# solve ocp in side process
# this is that function
# no it can't be a control loop then
# the actual control loop is trajectory following,
# but it continuously checks whether it can get the new path
# the path needs to be time-index because it's solved on old data
# (it takes time to solve the problem, but the horizon is then longer than thing).

# actually, before that, try solving the mpc at 500Hz with a short horizon
# and a small number of iterations, that might be feasible too,
# we need to see, there's no way to know.
# but doing it that way is certainly much easier to implement
# and probably better.
# i'm pretty sure that's how croco devs & mim_people do mpc


def CrocoIKMPCControlLoop(args, robot: RobotManager, solver, goal, i, past_data):
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    # check for goal
    if robot.robot_name == "yumi":
        goal_left, goal_right = goal
    SEerror = robot.getT_w_e().actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector
    if np.linalg.norm(err_vector) < robot.args.goal_error:
        #      print("Convergence achieved, reached destionation!")
        breakFlag = True

    # set initial state from sensor
    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    solver.problem.x0 = x0
    # warmstart solver with previous solution
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x0
    us_init = list(solver.us[1:]) + [solver.us[-1]]

    solver.solve(xs_init, us_init, args.max_solver_iter)
    xs = np.array(solver.xs)
    us = np.array(solver.us)
    vel_cmds = xs[1, robot.model.nq :]
    robot.sendQd(vel_cmds)

    log_item["qs"] = x0[: robot.model.nq]
    log_item["dqs"] = x0[robot.model.nv :]
    log_item["dqs_cmd"] = vel_cmds
    log_item["u_tau"] = us[0]

    return breakFlag, save_past_dict, log_item


def CrocoIKMPC(args, robot, goal, run=True):
    """
    IKMPC
    -----
    run mpc for a point-to-point inverse kinematics.
    note that the actual problem is solved on
    a dynamics level, and velocities we command
    are actually extracted from the state x(q,dq)
    """
    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    problem = createCrocoIKOCP(args, robot, x0, goal)
    if args.solver == "boxfddp":
        solver = crocoddyl.SolverBoxFDDP(problem)
    if args.solver == "csqp":
        solver = mim_solvers.SolverCSQP(problem)

    # technically should be done in controlloop because now
    # it's solved 2 times before the first command,
    # but we don't have time for details rn
    xs_init = [x0] * (solver.problem.T + 1)
    us_init = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.solve(xs_init, us_init, args.max_solver_iter)

    controlLoop = partial(CrocoIKMPCControlLoop, args, robot, solver)
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nq),
        "dqs_cmd": np.zeros(robot.model.nv),  # we're assuming full actuation here
        "u_tau": np.zeros(robot.model.nv),
    }
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager


def CrocoEndEffectorPathFollowingMPCControlLoop(
    args,
    robot: RobotManager,
    solver: crocoddyl.SolverBoxFDDP,
    path_planner: ProcessManager,
    i,
    past_data,
):
    """
    CrocoPathFollowingMPCControlLoop
    -----------------------------
    end-effector(s) follow their path(s).

    path planner can either be a function which spits out a list of path points
    or an instance of ProcessManager which spits out path points
    by calling ProcessManager.getData()
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    p = T_w_e.translation[:2]

    # NOTE: it's pointless to recalculate the path every time
    # if it's the same 2D path

    if type(path_planner) == types.FunctionType:
        pathSE3 = path_planner(T_w_e, i)
    else:
        path_planner.sendCommand(p)
        data = path_planner.getData()
        if data == None:
            if args.debug_prints:
                print("CTRL: got no path so not i won't move")
            robot.sendQd(np.zeros(robot.model.nv))
            log_item["qs"] = q.reshape((robot.model.nq,))
            log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
            return breakFlag, save_past_dict, log_item

        if data == "done":
            breakFlag = True

        path_pol, path2D_untimed = data
        path2D_untimed = np.array(path2D_untimed).reshape((-1, 2))
        # who cares if the velocity is right,
        # it should be kinda right so that we need less ocp iterations
        # and that's it
        max_base_v = np.linalg.norm(robot.model.velocityLimit[:2])
        path2D = path2D_timed(args, path2D_untimed, max_base_v)

        # create a 3D reference out of the path
        pathSE3 = path2D_to_SE3(path2D, args.handlebar_height)

    # TODO: EVIL AND HAS TO BE REMOVED FROM HERE
    if args.visualize_manipulator:
        if i % 20 == 0:
            robot.visualizer_manager.sendCommand({"frame_path": pathSE3})

    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    solver.problem.x0 = x0
    # warmstart solver with previous solution
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x0
    us_init = list(solver.us[1:]) + [solver.us[-1]]

    for i, runningModel in enumerate(solver.problem.runningModels):
        runningModel.differential.costs.costs[
            "gripperPose" + str(i)
        ].cost.residual.reference = pathSE3[i]
        # runningModel.differential.costs.costs['gripperPose'].cost.residual.reference = pathSE3[i]

    # idk if that's necessary
    solver.problem.terminalModel.differential.costs.costs[
        "gripperPose" + str(args.n_knots)
    ].cost.residual.reference = pathSE3[-1]
    # solver.problem.terminalModel.differential.costs.costs['gripperPose'].cost.residual.reference = pathSE3[-1]

    solver.solve(xs_init, us_init, args.max_solver_iter)
    xs = np.array(solver.xs)
    us = np.array(solver.us)
    vel_cmds = xs[1, robot.model.nq :]

    robot.sendQd(vel_cmds)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    return breakFlag, save_past_dict, log_item


def CrocoEndEffectorPathFollowingMPC(args, robot, x0, path_planner):
    """
    CrocoEndEffectorPathFollowingMPC
    -----
    run mpc for a point-to-point inverse kinematics.
    note that the actual problem is solved on
    a dynamics level, and velocities we command
    are actually extracted from the state x(q,dq).
    """

    problem = createCrocoEEPathFollowingOCP(args, robot, x0)
    if args.solver == "boxfddp":
        solver = crocoddyl.SolverBoxFDDP(problem)
    if args.solver == "csqp":
        solver = mim_solvers.SolverCSQP(problem)

    # technically should be done in controlloop because now
    # it's solved 2 times before the first command,
    # but we don't have time for details rn
    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    xs_init = [x0] * (solver.problem.T + 1)
    us_init = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.solve(xs_init, us_init, args.max_solver_iter)

    controlLoop = partial(
        CrocoEndEffectorPathFollowingMPCControlLoop, args, robot, solver, path_planner
    )
    log_item = {"qs": np.zeros(robot.model.nq), "dqs": np.zeros(robot.model.nv)}
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()


def BaseAndEEPathFollowingMPCControlLoop(
    args,
    robot: RobotManager,
    solver: crocoddyl.SolverBoxFDDP,
    path_planner: ProcessManager,
    grasp_pose,
    goal_transform,
    iter_n,
    past_data,
):
    """
    CrocoPathFollowingMPCControlLoop
    -----------------------------
    end-effector(s) follow their path(s).

    path planner can either be a function which spits out a list of path points
    or an instance of ProcessManager which spits out path points
    by calling ProcessManager.getData()
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    p = q[:2]
    # NOTE: this is the actual position, not what the path suggested
    # whether this or path reference should be put in is debateable.
    # this feels more correct to me.
    save_past_dict["path2D_untimed"] = p
    path_planner.sendCommand(p)

    ###########################
    #  get path from planner  #
    ###########################
    # NOTE: it's pointless to recalculate the path every time
    # if it's the same 2D path
    # get the path from the base from the current base position onward.
    # but we're still fast so who cares
    data = path_planner.getData()
    if data == None:
        if args.debug_prints:
            print("CTRL: got no path so not i won't move")
        robot.sendQd(np.zeros(robot.model.nv))
        log_item["qs"] = q.reshape((robot.model.nq,))
        log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
        return breakFlag, save_past_dict, log_item

    if data == "done":
        breakFlag = True
        robot.sendQd(np.zeros(robot.model.nv))
        log_item["qs"] = q.reshape((robot.model.nq,))
        log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
        return breakFlag, save_past_dict, log_item

    ##########################################
    #  construct timed 2D path for the base  #
    ##########################################
    path_pol_base, path2D_untimed_base = data
    path2D_untimed_base = np.array(path2D_untimed_base).reshape((-1, 2))
    # ideally should be precomputed somewhere
    max_base_v = np.linalg.norm(robot.model.velocityLimit[:2])
    # base just needs timing on the path
    path_base = path2D_timed(args, path2D_untimed_base, max_base_v)
    # and it's of height 0 (i.e. the height of base's planar joint)
    path_base = np.hstack((path_base, np.zeros((len(path_base), 1))))

    ###################################################
    #  construct timed SE3 path for the end-effector  #
    ###################################################
    # this works as follow
    # 1) find the previous path point of arclength base_to_handlebar_preferred_distance.
    # first part of the path is from there to current base position,
    # second is just the current base's plan.
    # 2) construct timing on the first part.
    # 3) join that with the already timed second part.
    # 4) turn the timed 2D path into an SE3 trajectory

    # NOTE: this can be O(1) instead of O(n) but i can't be bothered
    path_arclength = np.linalg.norm(p - past_data["path2D_untimed"])
    handlebar_path_index = -1
    for i in range(-2, -1 * len(past_data["path2D_untimed"]), -1):
        if path_arclength > args.base_to_handlebar_preferred_distance:
            handlebar_path_index = i
            break
        path_arclength += np.linalg.norm(
            past_data["path2D_untimed"][i - 1] - past_data["path2D_untimed"][i]
        )
    # i shouldn't need to copy-paste everything but what can you do
    path2D_handlebar_1_untimed = np.array(past_data["path2D_untimed"])
    # NOTE: BIG ASSUMPTION
    # let's say we're computing on time, and that's the time spacing
    # of previous path points.
    # this means you need to lower the control frequency argument
    # if you're not meeting deadlines.
    # if you really need timing information, you should actually
    # get it from ControlLoopManager instead of i,
    # but let's say this is better because you're forced to know
    # how fast you are instead of ducktaping around delays.
    # TODO: actually save timing, pass t instead of i to controlLoops
    # from controlLoopManager
    # NOTE: this might not working when rosified so check that first
    time_past = np.linspace(
        0.0, args.past_window_size * robot.dt, args.past_window_size
    )
    s = np.linspace(0.0, args.n_knots * args.ocp_dt, args.n_knots)
    path2D_handlebar_1 = np.hstack(
        (
            np.interp(s, time_past, path2D_handlebar_1_untimed[:, 0]).reshape((-1, 1)),
            np.interp(s, time_past, path2D_handlebar_1_untimed[:, 1]).reshape((-1, 1)),
        )
    )

    pathSE3_handlebar = path2D_to_SE3(path2D_handlebar_1, args.handlebar_height)
    pathSE3_handlebar_left = []
    pathSE3_handlebar_right = []
    for pathSE3 in pathSE3_handlebar:
        pathSE3_handlebar_left.append(goal_transform.act(pathSE3))
        pathSE3_handlebar_right.append(goal_transform.inverse().act(pathSE3))

    if args.visualize_manipulator:
        if iter_n % 20 == 0:
            robot.visualizer_manager.sendCommand({"path": path_base})
            robot.visualizer_manager.sendCommand({"frame_path": pathSE3_handlebar})

    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    solver.problem.x0 = x0
    # warmstart solver with previous solution
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x0
    us_init = list(solver.us[1:]) + [solver.us[-1]]

    for i, runningModel in enumerate(solver.problem.runningModels):
        # print('adding base', path_base[i])
        # print("this was the prev ref", runningModel.differential.costs.costs['base_translation' + str(i)].cost.residual.reference)
        runningModel.differential.costs.costs[
            "base_translation" + str(i)
        ].cost.residual.reference = path_base[i]
        if robot.robot_name != "yumi":
            runningModel.differential.costs.costs[
                "ee_pose" + str(i)
            ].cost.residual.reference = pathSE3_handlebar[i]
        else:
            runningModel.differential.costs.costs[
                "l_ee_pose" + str(i)
            ].cost.residual.reference = pathSE3_handlebar_left[i]
            runningModel.differential.costs.costs[
                "r_ee_pose" + str(i)
            ].cost.residual.reference = pathSE3_handlebar_right[i]

    # idk if that's necessary
    solver.problem.terminalModel.differential.costs.costs[
        "base_translation" + str(args.n_knots)
    ].cost.residual.reference = path_base[-1]
    if robot.robot_name != "yumi":
        solver.problem.terminalModel.differential.costs.costs[
            "ee_pose" + str(args.n_knots)
        ].cost.residual.reference = pathSE3_handlebar[-1]
    else:
        solver.problem.terminalModel.differential.costs.costs[
            "l_ee_pose" + str(args.n_knots)
        ].cost.residual.reference = pathSE3_handlebar[-1]
        solver.problem.terminalModel.differential.costs.costs[
            "r_ee_pose" + str(args.n_knots)
        ].cost.residual.reference = pathSE3_handlebar[-1]

    solver.solve(xs_init, us_init, args.max_solver_iter)
    xs = np.array(solver.xs)
    us = np.array(solver.us)
    vel_cmds = xs[1, robot.model.nq :]

    robot.sendQd(vel_cmds)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    return breakFlag, save_past_dict, log_item


def BaseAndEEPathFollowingMPC(args, robot, path_planner):
    """
    CrocoEndEffectorPathFollowingMPC
    -----
    run mpc for a point-to-point inverse kinematics.
    note that the actual problem is solved on
    a dynamics level, and velocities we command
    are actually extracted from the state x(q,dq).
    """

    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    problem = createBaseAndEEPathFollowingOCP(args, robot, x0)
    if args.solver == "boxfddp":
        solver = crocoddyl.SolverBoxFDDP(problem)
    if args.solver == "csqp":
        solver = mim_solvers.SolverCSQP(problem)

    xs_init = [x0] * (solver.problem.T + 1)
    us_init = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.solve(xs_init, us_init, args.max_solver_iter)

    controlLoop = partial(
        BaseAndEEPathFollowingMPCControlLoop, args, robot, solver, path_planner
    )
    log_item = {"qs": np.zeros(robot.model.nq), "dqs": np.zeros(robot.model.nv)}
    # TODO: put ensurance that save_past is not too small for this
    # or make a specific argument for THIS past-moving-window size
    # this is the end-effector's reference, so we should initialize that
    # TODO: verify this initialization actually makes sense
    T_w_e = robot.getT_w_e()
    save_past_dict = {"path2D_untimed": T_w_e.translation[:2]}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()

import casadi
import pinocchio as pin
if int(pin.__version__[0]) < 3:
    print("you need to have pinocchio version 3.0.0 or greater to use pinocchio.casadi!")
    exit()
from pinocchio import casadi as cpin
import numpy as np
from ur_simple_control.managers import RobotManager, ControlLoopManager
from ur_simple_control.util.encapsulating_ellipses import computeEncapsulatingEllipses, visualizeEllipses
import pickle
from importlib.resources import files
from types import SimpleNamespace
from os import path

"""
CREDIT: jrnh2023 pinocchio.casadi tutorial
Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   sum_t || q - robot.q0 ||**2
so that
      h(q) = target
      forall obstacles o,    (e_p - e_c)' e_A (e_p-e_c) >= 1
      TODO: here also define the table/floor as a plane, which would be 
                    normal_plane_vector (0,0,1) \cdot T_w_e.translation > 0
with h(q) the forward geometry (position of end effector) to be at target position,
e_A,e_c the ellipse matrix and center in the attached joint frame e_, and e_p = oMe^-1 o_p
the position of the obstacle point p in frame e_.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- pinocchio.casadi for writing the problem and computing its derivatives
- the IpOpt solver wrapped in casadi
- the meshcat viewer

It assumes that the ellipses parameters are already computed, see ellipses.py for that.
"""

# TODO: finish and verify
# make a separate function for a single ellipse based on the namespace
# if necessary save more info from solver
# --> memory is cheaper than computation,
#     don't save something only if it makes the struct confusing
def getSelfCollisionObstacles(args, robot : RobotManager):
    ellipses_path = files('ur_simple_control.robot_descriptions.ellipses').joinpath("ur5e_robotiq_ellipses.pickle")
    if path.exists(ellipses_path):
        file = open(ellipses_path, 'rb')
        ellipses = pickle.load(file)
        file.close()
    else:
        ellipses = computeEncapsulatingEllipses(args, robot)

    for ellipse in ellipses:
        ellipse.id = robot.model.getJointId(ellipse.name)
        l, P = np.linalg.eig(ellipse.A)
        ellipse.radius = 1 / l**0.5
        ellipse.rotation = P

    # TODO: move from here, you want this updated.
    # so it should be handled by some
    #if args.visualize_ellipses:
    #    visualizeEllipses(args, robot, ellipses)

    return ellipses

def createCasadiIKObstacleAvoidanceOCP(args, robot : RobotManager, T_goal: pin.SE3):
    # casadi equivalents of core pinocchio classes (used for everything)
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()
    nq = robot.model.nq
    nv = robot.model.nv
    q0 = robot.getQ()
    v0 = robot.getQd()
    # casadi symbolic variables for joint angles q
    cq = casadi.SX.sym("q", robot.model.nq, 1)
    cv = casadi.SX.sym("v", robot.model.nv, 1)
    # generate forward kinematics in casadi
    cpin.framesForwardKinematics(cmodel, cdata, cq)
    
    # kinematics constraint
    cnext = casadi.Function(
        "next",
        [cq, cv],
        [cpin.integrate(cmodel, cq, cv * args.ocp_dt)]
        )

    # error to goal function
    error6_tool = casadi.Function(
        "etool",
        [cq],
        [cpin.log6(cdata.oMf[robot.ee_frame_id].inverse() * cpin.SE3(T_goal)).vector],
    )


    # TODO: obstacles
    # here you need to define other obstacles, namely the table (floor) 
    # it's obviously going to be a plane 
    # alternatively just forbid z-axis of end-effector to be negative
    obstacles_sphere = [
        SimpleNamespace(radius=0.05, pos=np.array([0.0, 0.3, 0.0 + s]), name=f"obs_{i_s}")
        for i_s, s in enumerate(np.arange(0.0, 0.5, 0.125))
    ]
    for obstacle in obstacles_sphere:
        robot.visualizer_manager.sendCommand({"obstacle_sphere" : [obstacle.radius, obstacle.pos]})

    # define the optimizer/solver
    opti = casadi.Opti()

    # one set of decision variables per knot (both states and control input)
    var_qs = [opti.variable(nq) for t in range(args.n_knots + 1)]
    var_vs = [opti.variable(nv) for t in range(args.n_knots)]

    # running costs - x**2 and u**2 - used more for regularization than anything else
    totalcost = 0
    for t in range(args.n_knots):
        # running
        totalcost += 1e-3 * args.ocp_dt * casadi.sumsqr(var_qs[t])
        totalcost += 1e-4 * args.ocp_dt * casadi.sumsqr(var_vs[t])
    # terminal cost
    totalcost += 1e4 * casadi.sumsqr(error6_tool(var_qs[args.n_knots]))


    # initial state constraints
    # TODO: idk if you need to input x0 in this way, maybe robot.getQ(d) is better?
    opti.subject_to(var_qs[0] == q0)
    opti.subject_to(var_vs[0] == v0)
    # kinematics constraints between every knot pair
    for t in range(args.n_knots):
        opti.subject_to(cnext(var_qs[t], var_vs[t]) == var_qs[t + 1])
        opti.subject_to(var_vs[t] <= np.ones(robot.model.nv) * robot.max_qd)
        opti.subject_to(var_vs[t] >= -1 * np.ones(robot.model.nv) * robot.max_qd)

    # obstacle avoidance (hard) constraints: no ellipse should intersect any of the obstacles
    ellipses = getSelfCollisionObstacles(args, robot)
    cpos = casadi.SX.sym("p", 3) # obstacle position in ellipse frame
    for ellipse in ellipses:
        ellipse.e_pos = casadi.Function(
            f"e{ellipse.name}", [cq, cpos], [cdata.oMi[ellipse.id].inverse().act(casadi.SX(cpos))]
        )
        for obstacle in obstacles_sphere:
            for q in var_qs:
                # obstacle position in ellipsoid (joint) frame
                #e_pos = e.e_pos(var_q, o.pos)
                e_pos = ellipse.e_pos(q, obstacle.pos)
                # pretend obstacle is a point, and then
                # do no intersect with the point
                # TODO: make different equations for different obstacles,
                # most importantly box
                opti.subject_to((e_pos - ellipse.center).T @ ellipse.A @ (e_pos - ellipse.center) >= 1)

    # now that the ocp has been transcribed as nlp,
    # solve it
    opti.minimize(totalcost)
    opti.minimize(totalcost)
    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)
    opti.solver("ipopt")  # set numerical backend
    opti.set_initial(var_qs[0], q0)
    try:
        sol = opti.solve_limited()
        #sol_q = opti.value(var_q)
        sol_qs = [opti.value(var_q) for var_q in var_qs]
        sol_vs = [opti.value(var_v) for var_v in var_vs]
    except:
        print("ERROR in convergence, plotting debug info.")
        #sol_q = opti.debug.value(var_q)
        sol_qs = [opti.debug.value(var_q) for var_q in var_qs]
        sol_vs = [opti.debug.value(var_v) for var_v in var_vs]

    reference = {'qs' : sol_qs, 'vs' : sol_vs, 'dt' : args.ocp_dt}
    return reference, opti

def createCasadiReachingObstacleAvoidanceDynamicsOCP(args, robot : RobotManager, x0, goal : pin.SE3):
    # state = [joint_positions, joint_velocities], i.e. x = [q, v].
    # the velocity dimension does not need to be the same as the positions,
    # ex. consider a differential drive robot: it covers [x,y,theta], but has [vx, vtheta] as velocity
    nx = robot.model.nq + robot.model.nv
    ndx = 2 * robot.model.nv
    x0 = np.concat((robot.getQ(), robot.getQd()))
    cx = casadi.SX.sym("x", nx, 1)
    # dx = [velocity, acceleration] = [v,a]
    # acceleration is the same dimension as velocity
    cdx = casadi.SX.sym("dx", robot.model.nv * 2, 1)
    # separate state for less cumbersome expressions (they're still the same casadi variables)
    cq = cx[:robot.model.nq]
    cv = cx[robot.model.nq:]
    # acceleration is the same dimension as velocity
    caq = casadi.SX.sym("a", robot.model.nv, 1)
    cpin.forwardKinematics(cmodel, cdata, cq, cv, caq)
    cpin.updateFramePlacements(cmodel, cdata)

    # dynamics constraint
    cnext = casadi.Function(
        "next",
        [cx, caq],
        [
            casadi.vertcat(
                # TODO: i'm not sure i need to have the acceleration update for the position
                # because i am integrating the velocity as well.
                # i guess this is the way to go because we can't first integrate
                # the velocity and then the position, but have to do both simultaneously
                cpin.integrate(cmodel, cq, cv * args.ocp_dt + caq * args.ocp_dt**2),
                cv + caq * args.ocp_dt
            )
        ],
    )

    # cost function - reaching goal
    error6_tool = casadi.Function(
                    "etool6", 
                    [cx], 
                    [cpin.log6(cdata.oMf[endEffector_ID].inverse() * cpin.SE3(Mtarget)).vector],)

    # TODO: obstacles
    # here you need to define other obstacles, namely the table (floor) 
    # it's obviously going to be a plane 
    # alternatively just forbid z-axis of end-effector to be negative
    obstacles = []
    # one set of decision variables per knot (both states and control input)
    var_xs = [opti.variable(nx) for t in range(args.n_knots + 1)]
    var_as = [opti.variable(nv) for t in range(args.n_knots)]

    # running costs - x**2 and u**2 - used more for regularization than anything else
    totalcost = 0
    for t in range(args.n_knots):
        # running
        totalcost += 1e-3 * args.ocp_dt * casadi.sumsqr(var_xs[t][nq:])
        totalcost += 1e-4 * args.ocp_dt * casadi.sumsqr(var_as[t])
    # terminal cost
    totalcost += 1e4 * casadi.sumsqr(error6_tool(var_xs[T]))
    
    # now we combine the components into the OCP
    opti = casadi.Opti()

    # initial state constraints
    # TODO: idk if you need to input x0 in this way, maybe robot.getQ(d) is better?
    opti.subject_to(var_xs[0][:nq] == x0[:robot.model.nq])
    opti.subject_to(var_xs[0][nq:] == x0[robot.model.nq:])
    # dynamics constraint between every knot pair
    for t in range(args.n_knots):
        opti.subject_to(cnext(var_xs[t], var_as[t]) == var_xs[t + 1])

    # obstacle avoidance (hard) constraints: no ellipse should intersect any of the obstacles
    ellipses = getSelfCollisionObstacles(args, robot)
    cpos = casadi.SX.sym("p", 3)
    for ellipse in ellipses:
        for obstacle in obstacles:
            for x in var_xs:
                # obstacle position in ellipsoid (joint) frame
                #e_pos = e.e_pos(var_q, o.pos)
                e_pos = ellipse.e_pos(x[:nq], obstacle.pos)
                opti.subject_to((e_pos - ellipse.center).T @ e.A @ (e_pos - e.center) >= 1)

    # now that the ocp has been transcribed as nlp,
    # solve it
    opti.minimize(totalcost)
    opti.minimize(totalcost)
    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)
    opti.solver("ipopt")  # set numerical backend
    opti.set_initial(var_qs[0], q0)
    try:
        sol = opti.solve_limited()
        #sol_q = opti.value(var_q)
        sol_xs = [opti.value(var_x) for var_x in var_xs]
        sol_as = [opti.value(var_a) for var_a in var_as]
    except:
        print("ERROR in convergence, plotting debug info.")
        #sol_q = opti.debug.value(var_q)
        sol_xs = [opti.debug.value(var_x) for var_x in var_xs]
        sol_as = [opti.debug.value(var_a) for var_a in var_as]
    # TODO: extract qs and vs from xs, put in as
    # TODO: write dynamic trajectory following
    reference = {'qs' : sol_qs, 'vs' : sol_vs, 'dt' : args.ocp_dt}
    return reference, opti
    return sol_xs, sol_as

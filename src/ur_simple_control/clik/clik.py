import pinocchio as pin
import numpy as np
import copy
import argparse
from functools import partial
from ur_simple_control.managers import ControlLoopManager, RobotManager, ProcessManager
import time
from qpsolvers import solve_qp
import argparse
import importlib
import proxsuite

if importlib.util.find_spec("shapely"):
    from ur_simple_control.path_generation.planner import (
        path2D_to_timed_SE3,
        pathPointFromPathParam,
    )


def getClikArgs(parser):
    """
    getClikArgs
    ------------
    Every clik-related magic number, flag and setting is put in here.
    This way the rest of the code is clean.
    Force filtering is considered a core part of these control loops
    because it's not necessarily the same as elsewhere.
    If you want to know what these various arguments do, just grep
    though the code to find them (but replace '-' with '_' in multi-word arguments).
    All the sane defaults are here, and you can change any number of them as an argument
    when running your program. If you want to change a drastic amount of them, and
    not have to run a super long commands, just copy-paste this function and put your
    own parameters as default ones.
    """
    # parser = argparse.ArgumentParser(description='Run closed loop inverse kinematics \
    #        of various kinds. Make sure you know what the goal is before you run!',
    #        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ####################
    #  clik arguments  #
    ####################
    parser.add_argument(
        "--goal-error",
        type=float,
        help="the final position error you are happy with",
        default=1e-2,
    )
    parser.add_argument(
        "--tikhonov-damp",
        type=float,
        help="damping scalar in tikhonov regularization",
        default=1e-3,
    )
    # TODO add the rest
    parser.add_argument(
        "--clik-controller",
        type=str,
        help="select which click algorithm you want",
        default="dampedPseudoinverse",
        choices=[
            "dampedPseudoinverse",
            "jacobianTranspose",
            "invKinmQP",
            "QPproxsuite",
        ],
    )

    ###########################################
    #  force sensing and feedback parameters  #
    ###########################################
    parser.add_argument(
        "--alpha",
        type=float,
        help="force feedback proportional coefficient",
        default=0.01,
    )
    parser.add_argument(
        "--beta", type=float, help="low-pass filter beta parameter", default=0.01
    )

    ###############################
    #  path following parameters  #
    ###############################
    parser.add_argument(
        "--max-init-clik-iterations",
        type=int,
        help="number of max clik iterations to get to the first point",
        default=10000,
    )
    parser.add_argument(
        "--max-running-clik-iterations",
        type=int,
        help="number of max clik iterations between path points",
        default=1000,
    )
    parser.add_argument(
        "--viz-test-path",
        action=argparse.BooleanOptionalAction,
        help="number of max clik iterations between path points",
        default=False,
    )

    return parser


#######################################################################
#                             controllers                             #
#######################################################################
"""
controllers general
-----------------------
really trully just the equation you are running.
ideally, everything else is a placeholder,
meaning you can swap these at will.
if a controller has some additional arguments,
those are put in via functools.partial in getClikController,
so that you don't have to worry about how your controller is handled in
the actual control loop after you've put it in getClikController,
which constructs the controller from an argument to the file.
"""


def dampedPseudoinverse(tikhonov_damp, J, err_vector):
    qd = (
        J.T
        @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * tikhonov_damp)
        @ err_vector
    )
    return qd


def jacobianTranspose(J, err_vector):
    qd = J.T @ err_vector
    return qd


# TODO: put something into q of the QP
# also, put in lb and ub
# this one is with qpsolvers
def invKinmQP(J, err_vector, lb=None, ub=None, past_qd=None):
    """
    invKinmQP
    ---------
    generic QP:
        minimize 1/2 x^T P x + q^T x
        subject to
                 G x \leq h
                 A x = b
                 lb <= x <= ub
    inverse kinematics QP:
        minimize 1/2 qd^T P qd
                    + q^T qd (optional secondary objective)
        subject to
                 G qd \leq h (optional)
                 J qd = b    (mandatory)
                 lb <= qd <= ub (optional)
    """
    P = np.eye(J.shape[1], dtype="double")
    # secondary objective is given via q
    # we set it to 0 here, but we should give a sane default here
    q = np.array([0] * J.shape[1], dtype="double")
    G = None
    b = err_vector
    A = J
    # TODO: you probably want limits here
    # lb = None
    # ub = None
    # lb *= 20
    # ub *= 20
    h = None
    # (n_vars, n_eq_constraints, n_ineq_constraints)
    # qp.init(H, g, A, b, C, l, u)
    # print(J.shape)
    # print(q.shape)
    # print(A.shape)
    # print(b.shape)
    # NOTE: you want to pass the previous solver, not recreate it every time
    ######################
    # solve it
    # qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="ecos")
    # qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog", verbose=True, initvals=np.ones(len(lb)))
    # if not (past_qd is None):
    #    qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="proxqp", verbose=False, initvals=past_qd)
    # else:
    #    qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="proxqp", verbose=False, initvals=J.T@err_vector)
    # qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="proxqp", verbose=True, initvals=0.01*J.T@err_vector)
    # qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog", verbose=False, initvals=0.01*J.T@err_vector)
    qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog", verbose=False)
    # qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="proxqp")
    return qd


def QPproxsuite(qp, lb, ub, J, err_vector):
    # proxsuite does lb <= Cx <= ub
    qp.settings.initial_guess = (
        proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
    )
    # qp.update(g=q, A=A, b=h, l=lb, u=ub)
    qp.update(A=J, b=err_vector)
    qp.solve()
    qd = qp.results.x

    if qp.results.info.status == proxsuite.proxqp.PROXQP_PRIMAL_INFEASIBLE:
        # if np.abs(qp.results.info.duality_gap) > 0.1:
        print("didn't solve shit")
        qd = None
    return qd


# TODO: calculate nice q (in QP) as the secondary objective
# this requires getting the forward kinematics hessian,
# a.k.a jacobian derivative w.r.t. joint positions  dJ/dq .
# the ways to do it are as follows:
#   1) shitty and sad way to do it by computing dJ/dq \cdot \dot{q}
#      with unit velocities (qd) and then stacking that (very easy tho)
#   2) there is a function in c++ pinocchio for it getKinematicsHessian and you could write the pybind
#   3) you can write it yourself following peter corke's quide (he has a tutorial on fwd kinm derivatives)
#   4) figure out what pin.computeForwardKinematicDerivatives and pin.getJointAccelerationDerivatives
#      actually do and use that
# HA! found it in a winter school
# use this
# and test it with finite differencing!
"""
class CostManipulability:
    def __init__(self,jointIndex=None,frameIndex=None):
        if frameIndex is not None:
            jointIndex = robot.model.frames[frameIndex].parent
        self.jointIndex = jointIndex if jointIndex is not None else robot.model.njoints-1
    def calc(self,q):
        J = self.J=pin.computeJointJacobian(robot.model,robot.data,q,self.jointIndex)
        return np.sqrt(det(J@J.T))
    def calcDiff(self,q):
        Jp = pinv(pin.computeJointJacobian(robot.model,robot.data,q,self.jointIndex))
        res = np.zeros(robot.model.nv)
        v0 = np.zeros(robot.model.nv)
        for k in range(6):
            pin.computeForwardKinematicsDerivatives(robot.model,robot.data,q,Jp[:,k],v0)
            JqJpk = pin.getJointVelocityDerivatives(robot.model,robot.data,self.jointIndex,pin.LOCAL)[0]
            res += JqJpk[k,:]
        res *= self.calc(q)
        return res

# use this as a starting point for finite differencing
def numdiff(func, x, eps=1e-6):
    f0 = copy.copy(func(x))
    xe = x.copy()
    fs = []
    for k in range(len(x)):
        xe[k] += eps
        fs.append((func(xe) - f0) / eps)
        xe[k] -= eps
    if isinstance(f0, np.ndarray) and len(f0) > 1:
        return np.stack(fs,axis=1)
    else:
        return np.matrix(fs)

# and here's example usage
# Tdiffq is used to compute the tangent application in the configuration space.
Tdiffq = lambda f,q: Tdiff1(f,lambda q,v:pin.integrate(robot.model,q,v),robot.model.nv,q)
c=costManipulability
Tg = costManipulability.calcDiff(q)
Tgn = Tdiffq(costManipulability.calc,q)
#assert( norm(Tg-Tgn)<1e-4)
"""


def QPManipMax(J, err_vector, lb=None, ub=None):
    """
    invKinmQP
    ---------
    generic QP:
        minimize 1/2 x^T P x + q^T x
        subject to
                 G x \leq h
                 A x = b
                 lb <= x <= ub
    inverse kinematics QP:
        minimize 1/2 qd^T P qd
                    + q^T qd (optional secondary objective)
        subject to
                 G qd \leq h (optional)
                 J qd = b    (mandatory)
                 lb <= qd <= ub (optional)
    """
    P = np.eye(J.shape[1], dtype="double")
    # secondary objective is given via q
    # we set it to 0 here, but we should give a sane default here
    q = np.array([0] * J.shape[1], dtype="double")
    G = None
    b = err_vector
    A = J
    # TODO: you probably want limits here
    # lb = None
    # ub = None
    h = None
    # qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="ecos")
    qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog")
    return qd


def getClikController(args, robot):
    """
    getClikController
    -----------------
    A string argument is used to select one of these.
    It's a bit ugly, bit totally functional and OK solution.
    we want all of theme to accept the same arguments, i.e. the jacobian and the error vector.
    if they have extra stuff, just map it in the beginning with partial
    NOTE: this could be changed to something else if it proves inappropriate later
    TODO: write out other algorithms
    """
    if args.clik_controller == "dampedPseudoinverse":
        return partial(dampedPseudoinverse, args.tikhonov_damp)
    if args.clik_controller == "jacobianTranspose":
        return jacobianTranspose
    # TODO implement and add in the rest
    # if controller_name == "invKinmQPSingAvoidE_kI":
    #    return invKinmQPSingAvoidE_kI
    # if controller_name == "invKinm_PseudoInv":
    #    return invKinm_PseudoInv
    # if controller_name == "invKinm_PseudoInv_half":
    #    return invKinm_PseudoInv_half
    if args.clik_controller == "invKinmQP":
        lb = -1 * np.array(robot.model.velocityLimit, dtype="double")
        # we do additional clipping
        lb = np.clip(lb, -1 * robot.max_qd, robot.max_qd)
        ub = np.array(robot.model.velocityLimit, dtype="double")
        ub = np.clip(ub, -1 * robot.max_qd, robot.max_qd)
        return partial(invKinmQP, lb=lb, ub=ub)
    if args.clik_controller == "QPproxsuite":
        H = np.eye(robot.model.nv)
        g = np.zeros(robot.model.nv)
        G = np.eye(robot.model.nv)
        A = np.eye(6, robot.model.nv)
        b = np.ones(6) * 0.1
        # proxsuite does lb <= Cx <= ub
        C = np.eye(robot.model.nv)
        lb = -1 * np.array(robot.model.velocityLimit, dtype="double")
        # we do additional clipping
        lb = np.clip(lb, -1 * robot.max_qd, robot.max_qd)
        ub = np.array(robot.model.velocityLimit, dtype="double")
        ub = np.clip(ub, -1 * robot.max_qd, robot.max_qd)
        qp = proxsuite.proxqp.dense.QP(robot.model.nv, 6, robot.model.nv)
        qp.init(H, g, A, b, G, lb, ub)
        qp.solve()
        return partial(QPproxsuite, qp, lb, ub)

    # if controller_name == "invKinmQPSingAvoidE_kI":
    #    return invKinmQPSingAvoidE_kI
    # if controller_name == "invKinmQPSingAvoidE_kM":
    #    return invKinmQPSingAvoidE_kM
    # if controller_name == "invKinmQPSingAvoidManipMax":
    #    return invKinmQPSingAvoidManipMax

    # default
    return partial(dampedPseudoinverse, args.tikhonov_damp)


def controlLoopClik(robot: RobotManager, clik_controller, i, past_data):
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    # first check whether we're at the goal
    SEerror = T_w_e.actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector
    if np.linalg.norm(err_vector) < robot.args.goal_error:
        breakFlag = True
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    # compute the joint velocities based on controller you passed
    # qd = clik_controller(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    qd = clik_controller(J, err_vector)
    if qd is None:
        print("the controller you chose didn't work, using dampedPseudoinverse instead")
        qd = dampedPseudoinverse(1e-2, J, err_vector)
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    log_item["err_norm"] = np.linalg.norm(err_vector).reshape((1,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    save_past_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    return breakFlag, save_past_item, log_item


def controlLoopClikDualArm(
    robot: RobotManager, clik_controller, goal_transform, i, past_data
):
    """
    controlLoopClikDualArm
    ---------------
    do point to point motion for each arm and its goal.
    that goal is generated from a single goal that you pass,
    and a transformation on the goal you also pass.
    the transformation is done in goal's frame (goal is and has
    to be an SE3 object).
    the transform is designed for the left arm,
    and its inverse is applied for the right arm.
    """
    breakFlag = False
    log_item = {}
    q = robot.getQ()
    T_w_e_left, T_w_e_right = robot.getT_w_e()
    #
    Mgoal_left = goal_transform.act(robot.Mgoal)
    Mgoal_right = goal_transform.inverse().act(robot.Mgoal)
    # print("robot.Mgoal", robot.Mgoal)
    # print("Mgoal_left", Mgoal_left)
    # print("Mgoal_right", Mgoal_right)

    SEerror_left = T_w_e_left.actInv(Mgoal_left)
    SEerror_right = T_w_e_right.actInv(Mgoal_right)

    err_vector_left = pin.log6(SEerror_left).vector
    err_vector_right = pin.log6(SEerror_right).vector

    if (np.linalg.norm(err_vector_left) < robot.args.goal_error) and (
        np.linalg.norm(err_vector_right) < robot.args.goal_error
    ):
        breakFlag = True
    J_left = pin.computeFrameJacobian(robot.model, robot.data, q, robot.l_ee_frame_id)
    J_right = pin.computeFrameJacobian(robot.model, robot.data, q, robot.r_ee_frame_id)
    # compute the joint velocities based on controller you passed
    # this works exactly the way you think it does:
    # the velocities for the other arm are 0
    # what happens to the base hasn't been investigated but it seems ok
    qd_left = clik_controller(J_left, err_vector_left)
    qd_right = clik_controller(J_right, err_vector_right)
    qd = qd_left + qd_right
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    return breakFlag, {}, log_item


# def clikDualArm(args, robot, goal, goal_transform, run=True):
#    """
#    clikDualArm
#    -----------
#    """
#    robot.Mgoal = copy.deepcopy(goal)
#    clik_controller = getClikController(args, robot)
#    controlLoop = partial(controlLoopClikDualArm, robot, clik_controller, goal_transform)
#    # we're not using any past data or logging, hence the empty arguments
#    log_item = {
#            'qs' : np.zeros(robot.model.nq),
#            'dqs' : np.zeros(robot.model.nv),
#            'dqs_cmd' : np.zeros(robot.model.nv),
#        }
#    save_past_dict = {}
#    loop_manager = ControlLoopManager(robot, controlLoop, args, save_past_dict, log_item)
#    if run:
#        loop_manager.run()
#    else:
#        return loop_manager


def controlLoopClikArmOnly(robot, clik_controller, i, past_data):
    breakFlag = False
    log_item = {}
    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    # first check whether we're at the goal
    SEerror = T_w_e.actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector
    if np.linalg.norm(err_vector) < robot.args.goal_error:
        breakFlag = True
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    # cut off base from jacobian
    J = J[:, 3:]
    # compute the joint velocities based on controller you passed
    qd = clik_controller(J, err_vector)
    # add the missing velocities back
    qd = np.hstack((np.zeros(3), qd))
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    return breakFlag, {}, log_item


def moveUntilContactControlLoop(
    args, robot: RobotManager, speed, clik_controller, i, past_data
):
    """
    moveUntilContactControlLoop
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    breakFlag = False
    # know where you are, i.e. do forward kinematics
    log_item = {}
    q = robot.getQ()
    # break if wrench is nonzero basically
    # wrench = robot.getWrench()
    # you're already giving the speed in the EE i.e. body frame
    # so it only makes sense to have the wrench in the same frame
    # wrench = robot._getWrenchInEE()
    wrench = robot.getWrench()
    # and furthermore it's a reasonable assumption that you'll hit the thing
    # in the direction you're going in.
    # thus we only care about wrenches in those direction coordinates
    mask = speed != 0.0
    # NOTE: contact getting force is a magic number
    # it is a 100% empirical, with the goal being that it's just above noise.
    # so far it's worked fine, and it's pretty soft too.
    if np.linalg.norm(wrench[mask]) > args.contact_detecting_force:
        print("hit with", np.linalg.norm(wrench[mask]))
        breakFlag = True
        robot.sendQd(np.zeros(robot.model.nq))
    if (args.pinocchio_only) and (i > 500):
        print("let's say you hit something lule")
        breakFlag = True
    # pin.computeJointJacobian is much different than the C++ version lel
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    # compute the joint velocities.
    qd = clik_controller(J, speed)
    robot.sendQd(qd)
    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["wrench"] = wrench.reshape((6,))
    return breakFlag, {}, log_item


def moveUntilContact(args, robot, speed):
    """
    moveUntilContact
    -----
    does clik until it feels something with the f/t sensor
    """
    assert type(speed) == np.ndarray
    clik_controller = getClikController(args, robot)
    controlLoop = partial(
        moveUntilContactControlLoop, args, robot, speed, clik_controller
    )
    # we're not using any past data or logging, hence the empty arguments
    log_item = {"wrench": np.zeros(6)}
    log_item["qs"] = np.zeros((robot.model.nq,))
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()
    print("Collision detected!!")


def moveL(args, robot: RobotManager, goal_point):
    """
    moveL
    -----
    does moveL.
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    # assert type(goal_point) == pin.pinocchio_pywrap.SE3
    robot.Mgoal = copy.deepcopy(goal_point)
    clik_controller = getClikController(args, robot)
    controlLoop = partial(controlLoopClik, robot, clik_controller)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "dqs_cmd": np.zeros(robot.model.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {
        "dqs_cmd": np.zeros(robot.model.nv),
    }
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()


def moveLDualArm(args, robot: RobotManager, goal_point, goal_transform, run=True):
    """
    moveL
    -----
    does moveL.
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    # assert type(goal_point) == pin.pinocchio_pywrap.SE3
    robot.Mgoal = copy.deepcopy(goal_point)
    clik_controller = getClikController(args, robot)
    controlLoop = partial(
        controlLoopClikDualArm, robot, clik_controller, goal_transform
    )
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "dqs_cmd": np.zeros(robot.model.nv),
    }
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    if run:
        loop_manager.run()
    else:
        return loop_manager


# TODO: implement
def moveLFollowingLine(args, robot, goal_point):
    """
    moveLFollowingLine
    ------------------
    make a path from current to goal position, i.e.
    just a straight line between them.
    the question is what to do with orientations.
    i suppose it makes sense to have one function that enforces/assumes
    that the start and end positions have the same orientation.
    then another version goes in a line and linearly updates the orientation
    as it goes
    """
    pass


def cartesianPathFollowingWithPlannerControlLoop(
    args, robot: RobotManager, path_planner: ProcessManager, i, past_data
):
    """
    cartesianPathFollowingWithPlanner
    -----------------------------
    end-effector(s) follow their path(s) according to what a 2D path-planner spits out
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    p = T_w_e.translation[:2]
    path_planner.sendFreshestCommand({"p": p})

    # NOTE: it's pointless to recalculate the path every time
    # if it's the same 2D path
    data = path_planner.getData()
    if data == None:
        if args.debug_prints:
            print("got no path so no doing anything")
        robot.sendQd(np.zeros(robot.model.nv))
        log_item["qs"] = q.reshape((robot.model.nq,))
        log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
        return breakFlag, save_past_dict, log_item
    if data == "done":
        breakFlag = True

    path_pol, path2D_untimed = data
    path2D_untimed = np.array(path2D_untimed).reshape((-1, 2))
    # should be precomputed somewhere but this is nowhere near the biggest problem
    max_base_v = np.linalg.norm(robot.model.robot.model.velocityLimit[:2])

    # 1) make 6D path out of [[x0,y0],...]
    # that represents the center of the cart
    pathSE3 = path2D_to_timed_SE3(args, path_pol, path2D_untimed, max_base_v)
    # print(path2D_untimed)
    # for pp in pathSE3:
    #    print(pp.translation)
    # TODO: EVIL AND HAS TO BE REMOVED FROM HERE
    if args.visualize_manipulator:
        robot.visualizer_manager.sendCommand({"path": pathSE3})

    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    # NOTE: obviously not path following but i want to see things working here
    SEerror = T_w_e.actInv(pathSE3[-1])
    err_vector = pin.log6(SEerror).vector
    lb = -1 * robot.model.robot.model.velocityLimit
    lb[1] = -0.001
    ub = robot.model.robot.model.velocityLimit
    ub[1] = 0.001
    # vel_cmd = invKinmQP(J, err_vector, lb=lb, ub=ub)
    vel_cmd = dampedPseudoinverse(0.002, J, err_vector)
    robot.sendQd(vel_cmd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    # time.sleep(0.01)
    return breakFlag, save_past_dict, log_item


def cartesianPathFollowingWithPlanner(
    args, robot: RobotManager, path_planner: ProcessManager
):
    controlLoop = partial(
        cartesianPathFollowingWithPlannerControlLoop, args, robot, path_planner
    )
    log_item = {"qs": np.zeros(robot.model.nq), "dqs": np.zeros(robot.model.nv)}
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()


def controlLoopCompliantClik(args, robot: RobotManager, i, past_data):
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}
    # know where you are, i.e. do forward kinematics
    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    wrench = robot.getWrench()
    # we need to overcome noise if we want to converge
    if np.linalg.norm(wrench) < args.minimum_detectable_force_norm:
        wrench = np.zeros(6)
    save_past_dict["wrench"] = copy.deepcopy(wrench)
    wrench = args.beta * wrench + (1 - args.beta) * np.average(
        np.array(past_data["wrench"]), axis=0
    )
    Z = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    wrench = Z @ wrench
    # first check whether we're at the goal
    SEerror = T_w_e.actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector
    if np.linalg.norm(err_vector) < robot.args.goal_error:
        breakFlag = True
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    # compute the joint velocities.
    # just plug and play different ones
    qd = (
        J.T
        @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * args.tikhonov_damp)
        @ err_vector
    )
    tau = J.T @ wrench
    # tau = tau[:6].reshape((6,1))
    qd += args.alpha * tau
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    # get actual velocity, not the commanded one.
    # although that would be fun to compare i guess
    # TODO: have both, but call the compute control signal like it should be
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["wrench"] = wrench.reshape((6,))
    log_item["tau"] = tau.reshape((robot.model.nv,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    return breakFlag, save_past_dict, log_item


# add a threshold for the wrench
def compliantMoveL(args, robot, goal_point, run=True):
    """
    compliantMoveL
    -----
    does compliantMoveL - a moveL, but with compliance achieved
    through f/t feedback
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    #    assert type(goal_point) == pin.pinocchio_pywrap.SE3
    robot.Mgoal = copy.deepcopy(goal_point)
    clik_controller = getClikController(args, robot)
    controlLoop = partial(controlLoopCompliantClik, args, robot)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "wrench": np.zeros(6),
        "tau": np.zeros(robot.model.nv),
        "dqs": np.zeros(robot.model.nv),
    }
    save_past_dict = {
        "wrench": np.zeros(6),
    }
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager


def clikCartesianPathIntoJointPath(
    args, robot, path, clikController, q_init, plane_pose
):
    """
    clikCartesianPathIntoJointPath
    ------------------------------
    functionality
    ------------
    Follows a provided Cartesian path,
    creates a joint space trajectory for it.
    Output format and timing works only for what the dmp code expects
    because it's only used there,
    and I never gave that code a lift-up it needs.

    return
    ------
    - joint_space_trajectory to follow the given path.

    arguments
    ----------
    - path: cartesian path given in task frame
    """
    # we don't know how many there will be, so a linked list is
    # clearly the best data structure here (instert is o(1) still,
    # and we aren't pressed on time when turning it into an array later)
    qs = []
    # let's use the functions we already have. to do so
    # we need to create a new RobotManager with arguments for simulation,
    # otherwise weird things will happen.
    # we keep all the other args intact
    sim_args = copy.deepcopy(args)
    sim_args.pinocchio_only = True
    sim_args.fast_simulation = True
    sim_args.plotter = False
    sim_args.visualizer = False
    sim_args.save_log = False  # we're not using sim robot outside of this
    sim_args.max_iterations = 10000  # more than enough
    sim_robot = RobotManager(sim_args)
    sim_robot.q = q_init.copy()
    sim_robot._step()
    for pose in path:
        moveL(sim_args, sim_robot, pose)
        # loop logs is a dict, dict keys list preserves input order
        new_q = sim_robot.q.copy()
        if args.viz_test_path:
            # look into visualize.py for details on what's available
            T_w_e = sim_robot.getT_w_e()
            robot.updateViz({"q": new_q, "T_w_e": T_w_e, "point": T_w_e.copy()})
            # time.sleep(0.005)
        qs.append(new_q[:6])
        # plot this on the real robot

    ##############################################
    #  save the obtained joint-space trajectory  #
    ##############################################
    qs = np.array(qs)
    # we're putting a dmp over this so we already have the timing ready
    # TODO: make this general, you don't want to depend on other random
    # arguments (make this one traj_time, then put tau0 = traj_time there
    t = np.linspace(0, args.tau0, len(qs)).reshape((len(qs), 1))
    joint_trajectory = np.hstack((t, qs))
    # TODO handle saving more consistently/intentionally
    # (although this definitely works right now and isn't bad, just mid)
    # os.makedir -p a data dir and save there, this is ugly
    # TODO: check if we actually need this and if not remove the saving
    # whatever code uses this is responsible to log it if it wants it,
    # let's not have random files around.
    np.savetxt("./joint_trajectory.csv", joint_trajectory, delimiter=",", fmt="%.5f")
    return joint_trajectory


def controlLoopClikDualArmsOnly(
    robot: RobotManager, clik_controller, goal_transform, i, past_data
):
    """
    controlLoopClikDualArmsOnly
    ---------------
    """
    breakFlag = False
    log_item = {}
    q = robot.getQ()
    T_w_e_left, T_w_e_right = robot.getT_w_e()
    #
    Mgoal_left = robot.Mgoal.act(goal_transform)
    Mgoal_right = robot.Mgoal.act(goal_transform.inverse())

    SEerror_left = T_w_e_left.actInv(Mgoal_left)
    SEerror_right = T_w_e_right.actInv(Mgoal_right)

    err_vector_left = pin.log6(SEerror_left).vector
    err_vector_right = pin.log6(SEerror_right).vector

    if (np.linalg.norm(err_vector_left) < robot.args.goal_error) and (
        np.linalg.norm(err_vector_right) < robot.args.goal_error
    ):
        breakFlag = True
    J_left = pin.computeFrameJacobian(robot.model, robot.data, q, robot.l_ee_frame_id)
    J_left = J_left[:, 3:]
    J_right = pin.computeFrameJacobian(robot.model, robot.data, q, robot.r_ee_frame_id)
    J_right = J_right[:, 3:]

    # compute the joint velocities based on controller you passed
    qd_left = clik_controller(J_left, err_vector_left)
    qd_right = clik_controller(J_right, err_vector_right)
    # lb, ub = (-0.05 * robot.model.robot.model.velocityLimit, 0.05 * robot.model.robot.model.velocityLimit)
    # qd_left = invKinmQP(J_left, err_vector_left, lb=lb[3:], ub=ub[3:])
    # qd_right = invKinmQP(J_right, err_vector_right, lb=lb[3:], ub=ub[3:])
    qd = qd_left + qd_right
    qd = np.hstack((np.zeros(3), qd))
    robot.sendQd(qd)

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dqs"] = robot.getQd().reshape((robot.model.nv,))
    log_item["dqs_cmd"] = qd.reshape((robot.model.nv,))
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    return breakFlag, {}, log_item


def moveLDualArmsOnly(args, robot: RobotManager, goal_point, goal_transform, run=True):
    """
    moveL
    -----
    does moveL.
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    # assert type(goal_point) == pin.pinocchio_pywrap.SE3
    robot.Mgoal = copy.deepcopy(goal_point)
    clik_controller = getClikController(args, robot)
    controlLoop = partial(
        controlLoopClikDualArmsOnly, robot, clik_controller, goal_transform
    )
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "dqs_cmd": np.zeros(robot.model.nv),
    }
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    if run:
        loop_manager.run()
    else:
        return loop_manager


if __name__ == "__main__":
    args = get_args()
    robot = RobotManager(args)
    Mgoal = robot.defineGoalPointCLI()
    clik_controller = getClikController(args, robot)
    controlLoop = partial(controlLoopClik, robot, clik_controller)
    # we're not using any past data or logging, hence the empty arguments
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, {})
    loop_manager.run()

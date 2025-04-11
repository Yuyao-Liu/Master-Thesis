from typing import Callable
import numpy as np
from qpsolvers import solve_qp

# TODO: if importlib.files ... to make it an optional import
from proxsuite import proxqp

from functools import partial
from argparse import Namespace

from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.robots.interfaces.dual_arm_interface import DualArmInterface


def getIKSolver(
    args: Namespace, robot: SingleArmInterface | DualArmInterface
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    getIKSolver
    -----------------
    A string argument is used to select one of these.
    It's a bit ugly, bit totally functional and OK solution.
    we want all of theme to accept the same arguments, i.e. the jacobian and the error vector.
    if they have extra stuff, just map it in the beginning with partial
    NOTE: this could be changed to something else if it proves inappropriate later
    TODO: write out other algorithms
    """
    if args.ik_solver == "dampedPseudoinverse":
        return partial(dampedPseudoinverse, args.tikhonov_damp)
    if args.ik_solver == "jacobianTranspose":
        return jacobianTranspose
    # TODO implement and add in the rest
    # if controller_name == "invKinmQPSingAvoidE_kI":
    #    return invKinmQPSingAvoidE_kI
    # if controller_name == "invKinm_PseudoInv":
    #    return invKinm_PseudoInv
    # if controller_name == "invKinm_PseudoInv_half":
    #    return invKinm_PseudoInv_half
    if args.ik_solver == "QPquadprog":
        lb = -1 * robot.max_v
        ub = robot.max_v
        return partial(QPquadprog, lb=lb, ub=ub)
    # via quadprog
    #    if args.ik_solver == "QPManipMax":
    #        lb = -1 * robot.max_v
    #        ub = robot.max_v
    #        return partial(QPManipMax, lb=lb, ub=ub)
    if args.ik_solver == "QPproxsuite":
        H = np.eye(robot.nv)
        g = np.zeros(robot.nv)
        G = np.eye(robot.nv)
        J = robot.getJacobian()
        A = np.eye(J.shape[0], robot.nv)
        b = np.ones(J.shape[0]) * 0.1
        qp = proxqp.dense.QP(robot.nv, J.shape[0], robot.nv)
        # proxqp does lb <= Cx <= ub
        C = np.eye(robot.nv)
        lb = -1 * robot.max_v
        ub = robot.max_v
        qp.init(H, g, A, b, G, lb, ub)
        qp.solve()
        return partial(QPproxsuite, qp)

    # if controller_name == "invKinmQPSingAvoidE_kI":
    #    return invKinmQPSingAvoidE_kI
    # if controller_name == "invKinmQPSingAvoidE_kM":
    #    return invKinmQPSingAvoidE_kM
    # if controller_name == "invKinmQPSingAvoidManipMax":
    #    return invKinmQPSingAvoidManipMax

    # default
    return partial(dampedPseudoinverse, args.tikhonov_damp)


def dampedPseudoinverse(
    tikhonov_damp: float, J: np.ndarray, err_vector: np.ndarray
) -> np.ndarray:
    qd = (
        J.T
        @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * tikhonov_damp)
        @ err_vector
    )
    return qd


def jacobianTranspose(J: np.ndarray, err_vector: np.ndarray) -> np.ndarray:
    qd = J.T @ err_vector
    return qd


# TODO: put something into q of the QP
# also, put in lb and ub
# this one is with qpsolvers
def QPquadprog(
    J: np.ndarray, V_e_e: np.ndarray, lb=None, ub=None, past_qd=None
) -> np.ndarray:
    """
    QPquadprog
    ---------
    generic QP:
        minimize 1/2 x^T P x + q^T x
        subject to
                 G x \\leq h
                 A x = b
                 lb <= x <= ub
    inverse kinematics QP:
        minimize 1/2 qd^T P qd
                    + q^T qd (optional secondary objective)
        subject to
                 G qd \\leq h (optional)
                 J qd = b    (mandatory)
                 lb <= qd <= ub (optional)
    """
    P = np.eye(J.shape[1], dtype="double")
    # secondary objective is given via q
    # we set it to 0 here, but we should give a sane default here
    q = np.array([0] * J.shape[1], dtype="double")
    G = None
    # NOTE: if err_vector is too big, J * qd = err_vector is infeasible
    # for the given ub, lb! (which makes perfect sense as it makes the constraints
    # incompatible)
    # thus we have to scale it to the maximum followable value under the given
    # inequality constraints on the velocities.
    # TODO:
    # unfortunatelly we need to do some non-trivial math to figure out what the fastest
    # possible end-effector velocity in the V_e_e direction is possible
    # given our current bounds. this is because it depends on the configuration,
    # i.e. on J of course - if we are in a singularity, then maximum velocity is 0!
    # and otherwise it's proportional to the manipulability ellipsoid.
    # NOTE: for now we're just eyeballing for sport
    # NOTE: this fails in low manipulability regions (which makes perfect sense also)
    V_e_e_norm = np.linalg.norm(V_e_e)
    max_V_e_e_norm = 0.3
    if V_e_e_norm < max_V_e_e_norm:
        b = V_e_e
    else:
        b = (V_e_e / V_e_e_norm) * max_V_e_e_norm
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


def QPproxsuite(
    qp: proxqp.dense.QP,
    J: np.ndarray,
    V_e_e: np.ndarray,
) -> np.ndarray:
    # proxqp does lb <= Cx <= ub
    qp.settings.initial_guess = proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
    # qp.update(g=q, A=A, b=h, l=lb, u=ub)
    # NOTE: if err_vector is too big, J * qd = err_vector is infeasible
    # for the given ub, lb! (which makes perfect sense as it makes the constraints
    # incompatible)
    # thus we have to scale it to the maximum followable value under the given
    # inequality constraints on the velocities.
    # TODO:
    # unfortunatelly we need to do some non-trivial math to figure out what the fastest
    # possible end-effector velocity in the V_e_e direction is possible
    # given our current bounds. this is because it depends on the configuration,
    # i.e. on J of course - if we are in a singularity, then maximum velocity is 0!
    # and otherwise it's proportional to the manipulability ellipsoid.
    # NOTE: for now we're just eyeballing for sport
    # NOTE: this fails in low manipulability regions (which makes perfect sense also)
    V_e_e_norm = np.linalg.norm(V_e_e)
    max_V_e_e_norm = 0.3
    if V_e_e_norm < max_V_e_e_norm:
        b = V_e_e
    else:
        b = (V_e_e / V_e_e_norm) * max_V_e_e_norm
    # qp.update(A=J, b=err_vector)
    qp.update(A=J, b=b)
    qp.solve()
    qd = qp.results.x

    if qp.results.info.status == proxqp.PROXQP_PRIMAL_INFEASIBLE:
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
class CostManipulability:
    def __init__(self, jointIndex=None, frameIndex=None):
        if frameIndex is not None:
            jointIndex = robot.model.frames[frameIndex].parent
        self.jointIndex = (
            jointIndex if jointIndex is not None else robot.model.njoints - 1
        )

    def calc(self, q):
        J = self.J = pin.computeJointJacobian(
            robot.model, robot.data, q, self.jointIndex
        )
        return np.sqrt(det(J @ J.T))

    def calcDiff(self, q):
        Jp = pinv(pin.computeJointJacobian(robot.model, robot.data, q, self.jointIndex))
        res = np.zeros(robot.model.nv)
        v0 = np.zeros(robot.model.nv)
        for k in range(6):
            pin.computeForwardKinematicsDerivatives(
                robot.model, robot.data, q, Jp[:, k], v0
            )
            JqJpk = pin.getJointVelocityDerivatives(
                robot.model, robot.data, self.jointIndex, pin.LOCAL
            )[0]
            res += JqJpk[k, :]
        res *= self.calc(q)
        return res


"""

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


def QPManipMax(
    J: np.ndarray,
    V_e_e: np.ndarray,
    secondary_objective_vec: np.ndarray,
    lb=None,
    ub=None,
) -> np.ndarray:
    """
    QPManipMAx
    ---------
    generic QP:
        minimize 1/2 x^T P x + q^T x
        subject to
                 G x \\leq h
                 A x = b
                 lb <= x <= ub
    inverse kinematics QP:
        minimize 1/2 qd^T P qd
                    + q^T qd (where q is the partial deriviative of the manipulability index w.r.t. q)
        subject to
                 G qd \\leq h (optional)
                 J qd = b    (mandatory)
                 lb <= qd <= ub (optional)
    """
    P = np.eye(J.shape[1], dtype="double")
    # secondary objective is given via q
    # we set it to 0 here, but we should give a sane default here
    q = secondary_objective_vec
    G = None
    V_e_e_norm = np.linalg.norm(V_e_e)
    max_V_e_e_norm = 0.2
    if V_e_e_norm < max_V_e_e_norm:
        b = V_e_e
    else:
        b = (V_e_e / V_e_e_norm) * max_V_e_e_norm
    A = J
    h = None
    qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog", verbose=False)
    return qd

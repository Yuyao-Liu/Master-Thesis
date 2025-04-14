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

def compute_manipulability(J):
    # print(np.linalg.det(J @ J.T))
    return np.sqrt(np.linalg.det(J @ J.T) + np.eye(J.shape[0], J.shape[0]) * 1e-6)

def add_bias_and_noise(vec):
    noise_std=0.01
    bias = np.random.uniform(low=-0.5, high=0.5, size=vec.shape)
    vec = np.asarray(vec)
    bias = np.asarray(bias) if isinstance(bias, (list, np.ndarray)) else bias
    noise = np.random.normal(loc=0.0, scale=noise_std, size=vec.shape)
    return vec + bias + noise

# Transform a velocity vector from the world frame to the end-effector (EE) frame
def transform_velocity_to_e(robot):
    """
    Transform a velocity vector from the world frame to the end-effector (EE) frame.

    Parameters:
    - T_w_e: SE3 transformation matrix (Pinocchio SE3 object), representing the pose of the end-effector in the world frame.
    - u_ref_w: 6D velocity vector [v_x, v_y, v_z, ω_x, ω_y, ω_z] in the world frame.

    Returns:
    - u_ref_e: 6D velocity vector in the EE frame.
    """
    # Extract the rotation matrix R_w_e (3x3)
    T_w_e = robot.getT_w_e()
    R_w_e = T_w_e.rotation  

    # Extract linear and angular velocity components in the world frame
    u_ref_w = robot.u_ref_w
    v_w = u_ref_w[:3]  # Linear velocity in the world frame
    omega_w = u_ref_w[3:]  # Angular velocity in the world frame

    # Transform velocities to the EE frame
    v_e = R_w_e.T @ v_w  # Linear velocity in the EE frame
    omega_e = R_w_e.T @ omega_w  # Angular velocity in the EE frame

    # Combine into a 6D velocity vector
    u_ref_e = np.hstack((v_e, omega_e))
    
    return u_ref_e

def compute_ee2basedistance(robot):
    q = robot.getQ()
    x_base = q[0]
    y_base = q[1]
    T_w_e = robot.getT_w_e()
    x_ee = T_w_e.translation[0]
    y_ee = T_w_e.translation[1]
    d = np.sqrt((x_base-x_ee)**2+(y_base-y_ee)**2)
    return d

def parking_base(q, target_pose):
    """
    Compute the linear velocity (v) and angular velocity (omega) for a differential drive robot.
    
    Parameters:
    - q: Robot state, where q[0] and q[1] represent the x and y coordinates, and q[2], q[3] are quaternion values.
    - target_pose: (x, y, theta) Target position in meters and radians.

    Returns:
    - qd: An array containing the computed velocities.
    """

    # Control gains (adjustable)
    k1 = 1.0  # Linear velocity gain
    k2 = 2.0  # Angular velocity gain
    k3 = 1.0  # Orientation error gain

    # Extract robot's current pose
    x_r, y_r, theta_r = (q[0], q[1], np.arctan2(q[3], q[2]))  
    x_t, y_t, theta_t = target_pose  

    # Compute the relative position between robot and target
    dx = x_r - x_t
    dy = y_r - y_t
    rho = np.hypot(dx, dy)  # Euclidean distance to the target

    # Calculate the target's bearing angle in the global frame
    theta_target = np.arctan2(dy, dx)

    # Compute the bearing error in the robot's local frame
    gamma = theta_target - theta_r + np.pi  # Adjust for orientation

    # Normalize the angle to (-pi, pi] range
    gamma = (gamma + np.pi) % (2 * np.pi) - np.pi

    # Compute the final orientation error
    delta = gamma + theta_r - theta_t
    delta = (delta + np.pi) % (2 * np.pi) - np.pi  # Normalize again

    # Compute linear velocity
    v = k1 * rho * np.cos(gamma)

    # Compute angular velocity, avoiding division by zero
    if gamma == 0:
        omega = 0  
    else:
        omega = k2 * gamma + k1 * (np.sin(gamma) * np.cos(gamma) / gamma) * (gamma + k3 * delta)

    # Construct the velocity output array
    qd = np.array([v, 0, omega, 0, 0, 0, 0, 0, 0])
    return qd


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


def keep_distance_nullspace(tikhonov_damp, q, J, err_vector, robot):
    J = np.delete(J, 1, axis=1)
    # q = add_bias_and_noise(q)
    (x_base, y_base, theta_base) = (q[0], q[1], np.arctan2(q[3], q[2]))
    T_w_e = robot.T_w_e
    (x_ee, y_ee) = (T_w_e.translation[0], T_w_e.translation[1])
    # J_w = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
    # joint_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # W = np.diag(joint_weights)

    # # Compute adaptive damping factor
    # manipulability = compute_manipulability(J[:, 2:])
    # lambda_adaptive = tikhonov_damp / manipulability
    # # Compute the weighted damped pseudo-inverse
    # JWJ = J @ np.linalg.inv(W) @ J.T + lambda_adaptive * np.eye(J.shape[0])
    # J_pseudo = np.linalg.inv(W) @ J.T @ np.linalg.inv(JWJ)
    
    J_pseudo = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * tikhonov_damp)
    # Compute primary task velocity
    qd_task = J_pseudo @ err_vector

    ### compute q_null ###
    d_target = 0.6  # Minimum allowed EE-base distance
    dx = x_ee - x_base
    dy = y_ee - y_base
    d_current = np.hypot(dx, dy)
    print(d_current)
    I = np.eye(J.shape[1])
    N = I - J_pseudo @ J
    
    Jx = J[0, :]
    Jy = J[1, :]
    Jbx = np.zeros_like(Jx)
    Jbx[0] = 1
    Jd = (dx * (Jx - Jbx) + dy * Jy)/d_current
    # z1 = -5 * Jd.T * np.sign(d_current - d_target)
    z1 = -20 * Jd.T * (d_current - d_target)
    # print(Jx,Jy)
    # print(z1)
    
    z2 = np.zeros_like(z1)
    
    
    # J_w = np.delete(J_w,1,axis=1)
    xd = J @ qd_task
    
    # qd = robot.getQd()
    # xd = J_w @ qd
    # print(xd)
    dir_vee = np.array([xd[0], xd[1]])
    dir_base = np.array([q[2], q[3]])
    dir_eb = np.array([dx, dy])
    
    ## b to a counterclockwise
    def angle_between_vectors(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        theta_a = np.arctan2(a[1], a[0])
        theta_b = np.arctan2(b[1], b[0])
        angle = theta_a - theta_b
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        # dot_product = np.dot(a, b)
        # dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # angle = np.arccos(dot_product)
        
        # cross_product = np.cross(a, b)
        
        # if cross_product < 0:
        #     angle = -angle
        if angle > np.pi/2:
            angle = angle - np.pi
        if angle < -np.pi/2:
            angle = angle + np.pi
        return angle
    dir_e_z = np.array([T_w_e.rotation[0, 2], T_w_e.rotation[0, 1]])
    theta = angle_between_vectors(dir_vee, dir_base)
    # theta = angle_between_vectors(dir_e_z, dir_base)
    z2[1] = 0.5 * (theta)
    # z2[2] = -z2[1]
    # print(z2[1])
    if np.abs(d_current - d_target) < 0.05:
        qd_null = N @ (z1 + z2)
        # qd_null = N @ z2
    else:
        qd_null = N @ z1
    # Combine primary task velocity and null space velocity
    return qd_task + qd_null

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

from robot_stuff.forw_kinm import *
#from anim_func import *
import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colr
import sys
import scipy.optimize
from qpsolvers import solve_qp
from qpsolvers import dense_solvers 
from scipy.linalg import sqrtm



# hardcoded for all joints
# of course each joint can have its own limit
def clampVelocity(del_thet):
    for indeks in range(len(del_thet)):
        if del_thet[indeks] > 3.0:
            del_thet[indeks] = 3.0 
        
        if del_thet[indeks] < -3.0:
            del_thet[indeks] = -3.0 
    return del_thet
# r is the Robot_raw
# t is the target position



def invKinm_Jac_T(r, t):
    e = t - r.p_e
    num = np.dot(e, r.jac_tri @ r.jac_tri.T @ e)
    den = np.dot(r.jac_tri @ r.jac_tri.T @ e, r.jac_tri @ r.jac_tri.T @ e)
    alpha = num / den
    del_thet = alpha * r.jac_tri.T @ e

# clamping for joint rotation limits
    del_thet = clampVelocity(del_thet)

# if you want a damping constant other than alpha
#        del_thet = 0.011 * r.jac_tri.T @ e

    return del_thet


# using the nullspace for the comfort function
# when doing manipulability, use the nullspace then go to vec_toward_greater_manip
def invKinm_PseudoInv(r, t):
    e = t - r.p_e

    psedo_inv = np.linalg.pinv(r.jac_tri)
    del_thet = psedo_inv @ e
# we can add any nulspace vector to del_thet
# and given the constraints, we should implement some sort of a comfort function
# the min and max theta for each angle are hardcoded, but that will be changed 
# they are hardcoded to +/- pi # 3/4 with center at 0
# thus for all i q_iM - q_im = pi * 6/4
# the added q_0 must be left multiplyed by (np.eye(n) - np.linalg.pinv(r.jac_tri) @ r.jac_tri)
# we take into account the current theta (punish more if closer to the limit)
# the formula is 3.57 in siciliano 
    theta_for_limits = []
    for k in range(r.ndof):
        theta_for_limits.append( (-1/r.ndof) * (r.joints[k].theta / (np.pi * 1.5)))
    theta_for_limits = np.array(theta_for_limits, dtype=np.float32)

    del_thet += (np.eye(r.ndof) - psedo_inv @ r.jac_tri) @ theta_for_limits

    del_thet = clampVelocity(del_thet)

    return del_thet

# what is this, i don't know
def invKinm_PseudoInv_half(r, t):
    e = t - r.p_e

    #psedo_inv = np.linalg.pinv(r.jac_tri)
    #psedo_inv = r.jac_tri.T @ sqrtm(r.jac_tri @ r.jac_tri.T)
    psedo_inv = r.jac_tri.T @ np.linalg.inv(sqrtm(r.jac_tri @ r.jac_tri.T))
    #psedo_inv =  sqrtm(r.jac_tri.T @ r.jac_tri) @ r.jac_tri.T
    del_thet = psedo_inv @ e
#    theta_for_limits = []
#    for k in range(r.ndof):
#        theta_for_limits.append( (-1/r.ndof) * (r.joints[k].theta / (np.pi * 1.5)))
#    theta_for_limits = np.array(theta_for_limits, dtype=np.float32)
#
#    del_thet += (np.eye(r.ndof) - psedo_inv @ r.jac_tri) @ theta_for_limits

    del_thet = clampVelocity(del_thet)

    return del_thet




# using the nullspace singularity avoidance
def invKinmSingAvoidance_PseudoInv(r, e):
#    e = t - r.p_e

    psedo_inv = np.linalg.pinv(r.jac_tri)
    del_thet = psedo_inv @ e
# we can add any nulspace vector to del_thet
# and given the constraints, we should implement some sort of a comfort function
# the min and max theta for each angle are hardcoded, but that will be changed 
# they are hardcoded to +/- pi # 3/4 with center at 0
# thus for all i q_iM - q_im = pi * 6/4
# the added q_0 must be left multiplyed by (np.eye(n) - np.linalg.pinv(r.jac_tri) @ r.jac_tri)
# we take into account the current theta (punish more if closer to the limit)
# the formula is 3.57 in siciliano 
    gradMtoE = r.calcMToEGradient_kM()
#    print(gradMtoE)

    del_thet += (np.eye(r.ndof) - psedo_inv @ r.jac_tri) @ gradMtoE

    del_thet = clampVelocity(del_thet)

    return del_thet




def invKinm_dampedSquares(r, t):
    e = t - r.p_e
#    e = np.array([0.0,0.0,-1.0], dtype=np.float32)
#    e = np.array([0.0,1.0,0.0], dtype=np.float32)
#    e = np.array([1.0,0.0,0.0], dtype=np.float32)
    lamda = 0.3
    iden = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=np.float32)

    del_thet = r.jac_tri.T @ np.linalg.inv(r.jac_tri @ r.jac_tri.T + lamda**2 * iden) @ e

    del_thet = clampVelocity(del_thet)

    # let's try to use the calculation which uses svd
    # the derivation is in the iksurvey section 6
    # the final equation is (11) and it calculates all left of e in above del_thet

    # something is wrong here and i have  no idea what
#    m = 3
#    n = len(r.jac_tri[0,:])
#    svdd = np.zeros((n, m))
#    svd = np.linalg.svd(r.jac_tri) # the output is a list of 3 matrices: U, D and V
#    # important note: D is not returned as a diagonal matrix, but as the list of diagonal entries
#    for s in range(m): # 3 is the maximum rank of jacobian
#        svdd = svdd + (svd[1][s] / (svd[1][s] ** 2 + lamda ** 2)) * svd[2][:,s].reshape(n,1) @ svd[0][:,s].reshape(1, m)


#        del_thet = svdd @ e
#        del_thet = clampVelocity(del_thet)

    return del_thet


def invKinmQP(r, t):
    P = np.eye(r.ndof, dtype="double")
    q = np.array([0] * r.ndof, dtype="double") # should be q imo
    #G = np.eye(r.ndof, dtype="double")
    G = None
    e = t - r.p_e
#        e = np.array([0.0, 1.0, 0.0])
    b = np.array(e, dtype="double")
    A = np.array(r.jac_tri, dtype="double")
    #lb = np.array([-3] * r.ndof, dtype="double")
    lb = None
    #ub = np.array([3] * r.ndof, dtype="double")
    ub = None
    #h = ub
    h = None
 
 
    del_thet = solve_qp(P, q, G, h, A, b, lb, ub, solver="ecos")
    #del_thet = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog")
 
    return del_thet





# qp formulation, solved with scipy
def invKinmGradDesc(r, t):
    
    def getEEPos(thetas, r, t):
        p_e = r.eePositionAfterForwKinm(thetas)
        e = t - p_e
        error = np.sqrt(np.dot(e,e))
        return error
    
    def toOptim(thetas):
        #return np.sqrt(np.dot(thetas, thetas))
        return np.dot(thetas, thetas)
    e = t - r.p_e
    lb = []
    ub = []

    def constraint(r, e):
        # jac_tri @ del_thet must be equal to e
        # when doing manipulability optimization it will be e + vec_toward_greater_manip
        return scipy.optimize.LinearConstraint(r.jac_tri, e, e)


    for bo in range(len(r.joints)):
        lb.append(-3.0)
        ub.append(3.0)
    bounds = scipy.optimize.Bounds(lb, ub)

    error = np.sqrt(np.dot(e,e))
    thetas_start = []
    for th in range(r.ndof):
        thetas_start.append(r.joints[th].theta)
    thetas_start = np.array(thetas_start, dtype=np.float32)

    lin_constraint = constraint(r, e)
    if (r.clamp == 1):
        res = scipy.optimize.minimize(toOptim, thetas_start, method='SLSQP', constraints=lin_constraint, bounds=bounds)
    else:
        res = scipy.optimize.minimize(toOptim, thetas_start, method='SLSQP', constraints=lin_constraint)
#        res = scipy.optimize.minimize(getEEPos, thetas_start, args=(r,t), method='SLSQP', constraints=lin_constraint, bounds=bounds)
#        res = scipy.optimize.minimize(toOptim, thetas_start, method='CG', bounds=bounds)
    # without constraints it returns some crazy big numbres like 10**300 or sth
    # so something is seriously wrong there
    del_thet = []
    for bla in range(len(res.x)):
        del_thet.append(float(res.x[bla]))
#            del_thet.append(res.x[bla] - 0.01)
#        for bla in range(len(res.x)):
#            del_thet.append(float(res.x[bla]))
#            del_thet[bla] += 0.01
#            print(del_thet[bla])
#        print("del_thet")
#        print(del_thet)

#        del_thet = np.array(del_thet, dtype=np.float32)
    del_thet = clampVelocity(del_thet)
    return del_thet



###############################################################
# IK with singularity avoidance via QP
###############################################################

# qp formulation, solved with scipy
def invKinmSingAvoidanceWithQP_kM(r, t):
    
    def getEEPos(thetas, r, t):
        p_e = r.eePositionAfterForwKinm(thetas)
        e = t - p_e
        error = np.sqrt(np.dot(e,e))
        return error
    
    # E is shere toward which the manipulability elipsoid M should move
    # the calculation is defined as a method in the robot_raw class
    def toOptim(thetas, r):
        # alpha is a coef to select the amount of moving toward E
#        aplha = 3.03
        grad_to_E = r.calcMToEGradient_kM()
#        return np.dot(thetas, thetas) #+ coef_to_E @ thetas
        return np.dot(thetas, thetas) + np.dot(grad_to_E, thetas)
    e = t - r.p_e
    lb = []
    ub = []
    def constraint(r, e):
        # jac_tri @ del_thet must be equal to e
        # when doing manipulability optimization it will be e + vec_toward_greater_manip
        return scipy.optimize.LinearConstraint(r.jac_tri, e, e)


    for bo in range(r.ndof):
        lb.append(-3.0)
        ub.append(3.0)
    bounds = scipy.optimize.Bounds(lb, ub)

    error = np.sqrt(np.dot(e,e))
    thetas_start = []
    for th in range(r.ndof):
        thetas_start.append(r.joints[th].theta)
    thetas_start = np.array(thetas_start, dtype=np.float32)

    lin_constraint = constraint(r, e)
    if (r.clamp == 1):
        res = scipy.optimize.minimize(toOptim, thetas_start, args=(r), method='SLSQP', constraints=lin_constraint, bounds=bounds)
    else:
        res = scipy.optimize.minimize(toOptim, thetas_start, args=(r), method='SLSQP', constraints=lin_constraint)
#        res = scipy.optimize.minimize(toOptim, thetas_start, args=(r), method='CG', constraints=lin_constraint)
#        res = scipy.optimize.minimize(getEEPos, thetas_start, args=(r,t), method='SLSQP', constraints=lin_constraint, bounds=bounds)
#        res = scipy.optimize.minimize(toOptim, thetas_start, method='CG', bounds=bounds)
    # without constraints it returns some crazy big numbres like 10**300 or sth
    # so something is seriously wrong there
    del_thet = []
    for bla in range(len(res.x)):
        del_thet.append(float(res.x[bla]))
#            del_thet.append(res.x[bla] - 0.01)
#        for bla in range(len(res.x)):
#            del_thet.append(float(res.x[bla]))
#            del_thet[bla] += 0.01
#            print(del_thet[bla])
#        print("del_thet")
#        print(del_thet)

#        del_thet = np.array(del_thet, dtype=np.float32)
#    print(del_thet)
    del_thet = clampVelocity(del_thet)
    return del_thet



def invKinmSingAvoidanceWithQP_kI(r, t):
    
    def getEEPos(thetas, r, t):
        p_e = r.eePositionAfterForwKinm(thetas)
        e = t - p_e
        error = np.sqrt(np.dot(e,e))
        return error
    
    # E is shere toward which the manipulability elipsoid M should move
    # the calculation is defined as a method in the robot_raw class
    def toOptim(thetas, r):
        # alpha is a coef to select the amount of moving toward E
#        aplha = 3.03
        grad_to_E = r.calcMToEGradient_kI()
#        return np.dot(thetas, thetas) #+ coef_to_E @ thetas
        return np.dot(thetas, thetas) + np.dot(grad_to_E, thetas)
    e = t - r.p_e
    lb = []
    ub = []
    def constraint(r, e):
        # jac_tri @ del_thet must be equal to e
        # when doing manipulability optimization it will be e + vec_toward_greater_manip
        return scipy.optimize.LinearConstraint(r.jac_tri, e, e)


    for bo in range(r.ndof):
        lb.append(-3.0)
        ub.append(3.0)
    bounds = scipy.optimize.Bounds(lb, ub)

    error = np.sqrt(np.dot(e,e))
    thetas_start = []
    for th in range(r.ndof):
        thetas_start.append(r.joints[th].theta)
    thetas_start = np.array(thetas_start, dtype=np.float32)

    lin_constraint = constraint(r, e)
    if (r.clamp == 1):
        res = scipy.optimize.minimize(toOptim, thetas_start, args=(r), method='SLSQP', constraints=lin_constraint, bounds=bounds)
    else:
        res = scipy.optimize.minimize(toOptim, thetas_start, args=(r), method='SLSQP', constraints=lin_constraint)
#        res = scipy.optimize.minimize(toOptim, thetas_start, args=(r), method='CG', constraints=lin_constraint)
#        res = scipy.optimize.minimize(getEEPos, thetas_start, args=(r,t), method='SLSQP', constraints=lin_constraint, bounds=bounds)
#        res = scipy.optimize.minimize(toOptim, thetas_start, method='CG', bounds=bounds)
    # without constraints it returns some crazy big numbres like 10**300 or sth
    # so something is seriously wrong there
    del_thet = []
    for bla in range(len(res.x)):
        del_thet.append(float(res.x[bla]))
#            del_thet.append(res.x[bla] - 0.01)
#        for bla in range(len(res.x)):
#            del_thet.append(float(res.x[bla]))
#            del_thet[bla] += 0.01
#            print(del_thet[bla])
#        print("del_thet")
#        print(del_thet)

#        del_thet = np.array(del_thet, dtype=np.float32)
#    print(del_thet)
    del_thet = clampVelocity(del_thet)
    return del_thet



def invKinmQPSingAvoidE_kI(r, t):
    P = np.eye(r.ndof, dtype="double")
#    q = 0.5 * np.array(r.calcMToEGradient_kI(), dtype="double")
    q = np.array(r.calcMToEGradient_kI(), dtype="double")
#    G = np.eye(r.ndof, dtype="double")
#    G = []
    G = None
    e = t - r.p_e
    b = np.array(e, dtype="double")
    A = np.array(r.jac_tri, dtype="double")
    #lb = np.array([-3] * r.ndof, dtype="double")
    lb = None
    #ub = np.array([3] * r.ndof, dtype="double")
    ub = None
#    h = ub 
    h = None
 
#    del_thet = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog")
    del_thet = solve_qp(P, q, G, h, A, b, lb, ub, solver="ecos")
 
    return del_thet



def invKinmQPSingAvoidE_kM(r, t):
    P = np.eye(r.ndof, dtype="double")
#    q = 0.5 * np.array(r.calcMToEGradient_kM(), dtype="double")
    q = np.array(r.calcMToEGradient_kM(), dtype="double")
    #G = np.eye(r.ndof, dtype="double")
    G = None
    e = t - r.p_e
#        e = np.array([0.0, 1.0, 0.0])
    b = np.array(e, dtype="double")
    A = np.array(r.jac_tri, dtype="double")
    #lb = np.array([-3] * r.ndof, dtype="double")
    #ub = np.array([3] * r.ndof, dtype="double")
    lb = None
    ub = None
    #h = ub
    h = None
 
 
    del_thet = solve_qp(P, q, G, h, A, b, lb, ub, solver="ecos")
 
    return del_thet



def invKinmQPSingAvoidManipMax(r, t):
    P = np.eye(r.ndof, dtype="double")
    q = np.array(r.calcManipMaxGrad(), dtype="double")
    G = None
    e = t - r.p_e
    b = np.array(e, dtype="double")
    A = np.array(r.jac_tri, dtype="double")
    lb = np.array([-3] * r.ndof, dtype="double")
    ub = np.array([3] * r.ndof, dtype="double")
    h = None
 
    del_thet = solve_qp(P, q, G, h, A, b, lb, ub, solver="ecos")
 
    return del_thet


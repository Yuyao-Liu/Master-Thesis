import numpy as np


def error_test(p_e, target):
    e = np.abs(target - p_e)
    if e[0] < 0.002 and e[1] < 0.002 and e[2] < 0.002:
        return True
    else:
        return False

def goal_distance(achieved_goal, goal):
    return np.linalg.norm(goal - achieved_goal)


def calculateManipulabilityIndex(robot):
    M = robot.jac_tri @ robot.jac_tri.T
    return np.sqrt(np.linalg.det(M))

def calculateSmallestManipEigenval(robot):
    M = robot.jac_tri @ robot.jac_tri.T
    diagonal_of_svd_of_M = np.linalg.svd(M)[1]
    return diagonal_of_svd_of_M[diagonal_of_svd_of_M.argmin()]

def calculatePerformanceMetrics(robot):
    M = robot.jac_tri @ robot.jac_tri.T
    diagonal_of_svd_of_M = np.linalg.svd(M)[1]
    singularity = 0 
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError as e:
        print("ROKNUH U SINGULARITET!!!!!!!!!!!")
        singularity = 1
    return {'manip_index': np.sqrt(np.linalg.det(M)),
            'smallest_eigenval': diagonal_of_svd_of_M[diagonal_of_svd_of_M.argmin()],
            'singularity':singularity }


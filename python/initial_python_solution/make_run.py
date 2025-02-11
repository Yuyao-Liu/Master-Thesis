from robot_stuff.InverseKinematics import InverseKinematicsEnv
from robot_stuff.drawing import *
from robot_stuff.inv_kinm import *
from robot_stuff.utils import *

import numpy as np

# starting off with random goals, who cares rn
# policy is one of the inverse kinematics control algoritms
# TODO (optional): make it follow a predefined trajectory
# TODO (optional): vectorize it if it isn't fast enough (at least you avoid dynamic allocations)
def makeRun(controller, ik_env, n_iters, robot_index):
    # we'll shove everything into lists, and have it ready made
    data = {
    "qs" : np.zeros((n_iters, ik_env.robots[robot_index].ndof)),
    "q_dots" : np.zeros((n_iters, ik_env.robots[robot_index].ndof)),
    "manip_ell_eigenvals" : np.zeros((n_iters, 3)) ,
    "manip_elip_svd_rots" : np.zeros((n_iters, 3, 3)),
    "p_es" : np.zeros((n_iters, 3)),
    "vs" : np.zeros((n_iters, 6)), # all links linear velocities # NOT USED ATM
    "dists_to_goal" : np.zeros(n_iters),
    "manip_indeces" : np.zeros(n_iters),
    }

    for i in range(n_iters):
        q_dots = controller(ik_env.robots[robot_index], ik_env.goal)
        ik_env.simpleStep(q_dots, 1.0, robot_index)

        thetas = np.array([joint.theta for joint in ik_env.robots[robot_index].joints])
        data['qs'][i] = thetas
        data['q_dots'][i] = q_dots
        # NOTE: this is probably not correct, but who cares
        data['vs'][i] = ik_env.robots[robot_index].jacobian @ q_dots
        M = ik_env.robots[robot_index].jac_tri @ ik_env.robots[robot_index].jac_tri.T
        manip_index = np.sqrt(np.linalg.det(M))
        data['manip_indeces'][i] = manip_index
        _, diag_svd, rot = np.linalg.svd(M)
        # TODO check this: idk if this or the square roots are eigenvalues 
        data["manip_ell_eigenvals"][i] = np.sqrt(diag_svd)
        data["manip_elip_svd_rots"][i] = rot
        smallest_eigenval = diag_svd[diag_svd.argmin()]
        dist_to_goal = np.linalg.norm(ik_env.robots[robot_index].p_e - ik_env.goal)
        data['dists_to_goal'][i] = dist_to_goal
        data['p_es'][i] = ik_env.robots[robot_index].p_e
    return data

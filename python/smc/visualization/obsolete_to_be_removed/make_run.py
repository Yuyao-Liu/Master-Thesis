from robot_stuff.InverseKinematics import InverseKinematicsEnv
from robot_stuff.drawing import *
from robot_stuff.inv_kinm import *
from robot_stuff.utils import *
from ur_simple_control.managers import RobotManager
import pinocchio as pin

import numpy as np

# starting off with random goals, who cares rn
# policy is one of the inverse kinematics control algoritms
# TODO (optional): make it follow a predefined trajectory
# TODO (optional): vectorize it if it isn't fast enough (at least you avoid dynamic allocations)
def makeRun(controller, ik_env, n_iters, robot_index):
    # we'll shove everything into lists, and have it ready made
    data = {
    "qs" : np.zeros((n_iters, ik_env.robots[robot_index].ndof)),
    "dqs" : np.zeros((n_iters, ik_env.robots[robot_index].ndof)),
    "manip_ell_eigenvals" : np.zeros((n_iters, 3)) ,
    "manip_elip_svd_rots" : np.zeros((n_iters, 3, 3)),
    "p_es" : np.zeros((n_iters, 3)),
    "vs" : np.zeros((n_iters, 6)), # all links linear velocities # NOT USED ATM
    "dists_to_goal" : np.zeros(n_iters),
    "manip_indeces" : np.zeros(n_iters),
    }

    for i in range(n_iters):
        dqs = controller(ik_env.robots[robot_index], ik_env.goal)
        ik_env.simpleStep(dqs, 1.0, robot_index)

        thetas = np.array([joint.theta for joint in ik_env.robots[robot_index].joints])
        data['qs'][i] = thetas
        data['dqs'][i] = dqs
        # NOTE: this is probably not correct, but who cares
        data['vs'][i] = ik_env.robots[robot_index].jacobian @ dqs
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

"""
loadRun
--------
- args and robot as standard practise 
- log_data is a dict of data you wanna plot
- some keys are expected:
    - qs
- and nothing is really done with the other ones
   --> TODO put other ones in a new tab
"""
def loadRun(args, robot, log_data):
    # some of these you need, while the others you can compute here
    # anything extra can be copied here and then plotted in an extra tab or something
    assert "qs" in log_data.keys()
    n_iters = log_data["qs"].shape[0]
    recompute_checker = {
        "qs" : False,
        "dqs" : False,
        "manip_ell_eigenvals" : False,
        "manip_elip_svd_rots" : False,
        "p_es" : False,
        "vs" : False,
        "dists_to_goal" : False,
        "manip_indeces" : False
        }
    # we construct what's not in log_dict for a nice default window
    if "dqs" not in log_data.keys():
        log_data["dqs"] = np.zeros((n_iters, robot.n_joints))
        recompute_checker["dqs"] = True
    if "manip_ell_eigenvals" not in log_data.keys():
        log_data["manip_ell_eigenvals"] = np.zeros((n_iters, 3))
        recompute_checker["manip_ell_eigenvals"] = True
    if "manip_elip_svd_rots" not in log_data.keys():
        log_data["manip_elip_svd_rots"] = np.zeros((n_iters, 3, 3))
        recompute_checker["manip_elip_svd_rots"] = True
    if "p_es" not in log_data.keys():
        log_data["p_es"] = np.zeros((n_iters, 3))
        recompute_checker["p_es"] = True
    if "vs" not in log_data.keys():
        log_data["vs"] = np.zeros((n_iters, 6)) # all links linear velocities # NOT USED
        recompute_checker["vs"] = True
    if "dists_to_goal" not in log_data.keys():
        log_data["dists_to_goal"] = np.zeros(n_iters) # this should be general error terms
        recompute_checker["dists_to_goal"] = True
    if "manip_indeces" not in log_data.keys():
        log_data["manip_indeces"] = np.zeros(n_iters)
        recompute_checker["manip_indeces"] = True

    for i in range(n_iters):
        # NOTE: this is probably not correct, but who cares
        pin.forwardKinematics(robot.model, robot.data, log_data['qs'][i])
        J = pin.computeJointJacobian(robot.model, robot.data,
                log_data['qs'][i], robot.JOINT_ID)
        # cut of angular velocities 'cos i don't need them for the manip ellipsoid
        # but TODO: check that it's the top ones and not the bottom ones
        J_lin = J[:3, :6]
        if recompute_checker['vs']:
            log_data['vs'][i] = J[:6] @ log_data['dqs'][i]
        # TODO fix if you want it, whatever atm man
        # TODO you're doubling fkine, dont
        if recompute_checker['p_es']:
            log_data['p_es'][i] = robot.getT_w_e().translation

        M = J_lin @ J_lin.T
        manip_index = np.sqrt(np.linalg.det(M))
        _, diag_svd, rot = np.linalg.svd(M)
        if recompute_checker['manip_indeces']:
            log_data['manip_indeces'][i] = manip_index
        # TODO check this: idk if this or the square roots are eigenvalues 
        if recompute_checker['manip_ell_eigenvals']:
            log_data["manip_ell_eigenvals"][i] = np.sqrt(diag_svd)
        if recompute_checker['manip_elip_svd_rots']:
            log_data["manip_elip_svd_rots"][i] = rot
        # maybe plot just this idk, here it is
        smallest_eigenval = diag_svd[diag_svd.argmin()]
        # TODO: make these general error terms
        # also TODO: put this in the autogenerate plots tab
        if recompute_checker['dists_to_goal']:
            #dist_to_goal = np.linalg.norm(ik_env.robots[robot_index].p_e - ik_env.goal)
            #log_data['dists_to_goal'][i] = dist_to_goal
            pass
    return log_data

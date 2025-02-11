import pinocchio as pin
import numpy as np
import copy
import argparse
from functools import partial
from ur_simple_control.managers import ControlLoopManager, RobotManager
import time
from qpsolvers import solve_qp
import argparse

# CHECK THESE OTHER
from spatialmath.base import rotz, skew, r2q, tr2angvec

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
    #parser = argparse.ArgumentParser(description='Run closed loop inverse kinematics \
    #        of various kinds. Make sure you know what the goal is before you run!',
    #        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ####################
    #  clik arguments  #
    ####################
    parser.add_argument('--goal-error', type=float, \
            help="the final position error you are happy with", default=1e-2)
    parser.add_argument('--tikhonov-damp', type=float, \
            help="damping scalar in tikhonov regularization", default=1e-3)
    # TODO add the rest
    parser.add_argument('--clik-controller', type=str, \
            help="select which click algorithm you want", \
            default='dampedPseudoinverse', choices=['dampedPseudoinverse', 'jacobianTranspose', 'invKinmQP'])

    ###########################################
    #  force sensing and feedback parameters  #
    ###########################################
    parser.add_argument('--alpha', type=float, \
            help="force feedback proportional coefficient", \
            default=0.01)
    parser.add_argument('--beta', type=float, \
            help="low-pass filter beta parameter", \
            default=0.01)

    ###############################
    #  path following parameters  #
    ###############################
    parser.add_argument('--max-init-clik-iterations', type=int, \
            help="number of max clik iterations to get to the first point", default=10000)
    parser.add_argument('--max-running-clik-iterations', type=int, \
            help="number of max clik iterations between path points", default=1000)
    
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
    qd = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * tikhonov_damp) @ err_vector
    return qd

def jacobianTranspose(J, err_vector):
    qd = J.T @ err_vector
    return qd

def invKinmQP(J, err_vector):
    # maybe a lower precision dtype is equally good, but faster?
    P = np.eye(J.shape[1], dtype="double")
    # TODO: why is q all 0?
    q = np.array([0] * J.shape[1], dtype="double")
    G = None
    # TODO: extend for orientation as well
    b = err_vector#[:3]
    A = J#[:3]
    # TODO: you probably want limits here
    lb = None
    ub = None
    h = None
    #qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="ecos")
    qd = solve_qp(P, q, G, h, A, b, lb, ub, solver="quadprog")
    return qd

def getClikController(args):
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
    #if controller_name == "invKinmQPSingAvoidE_kI":
    #    return invKinmQPSingAvoidE_kI
    #if controller_name == "invKinm_PseudoInv":
    #    return invKinm_PseudoInv
    #if controller_name == "invKinm_PseudoInv_half":
    #    return invKinm_PseudoInv_half
    if args.clik_controller == "invKinmQP":
        return invKinmQP
    #if controller_name == "invKinmQPSingAvoidE_kI":
    #    return invKinmQPSingAvoidE_kI
    #if controller_name == "invKinmQPSingAvoidE_kM":
    #    return invKinmQPSingAvoidE_kM
    #if controller_name == "invKinmQPSingAvoidManipMax":
    #    return invKinmQPSingAvoidManipMax

    # default
    return partial(dampedPseudoinverse, args.tikhonov_damp)


def controlLoopClik(robot : RobotManager, clik_controller, i, past_data):
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    breakFlag = False
    log_item = {}
    q = robot.getQ()
    T_w_e = robot.getT_w_e()
    # first check whether we're at the goal
    SEerror = T_w_e.actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector 
    if np.linalg.norm(err_vector) < robot.args.goal_error:
#      print("Convergence achieved, reached destionation!")
        breakFlag = True
    #J = pin.computeJointJacobian(robot.model, robot.data, q, robot.JOINT_ID)
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id)
    # compute the joint velocities based on controller you passed
    qd = clik_controller(J, err_vector)
    robot.sendQd(qd)
    
    log_item['qs'] = q.reshape((robot.model.nq,))
    log_item['dqs'] = robot.getQd().reshape((robot.model.nq,))
    # we're not saving here, but need to respect the API, 
    # hence the empty dict
    return breakFlag, {}, log_item
    
    
def controlLoop_WholeBodyClik(robot : RobotManager, clik_controller, i, past_data):
    """
    controlLoop_WholeBodyClik
    ---------------
    generic control loop for clik for mobile robot
    """
    th = 10^-3
    if e_norm > th: # trajectory_planner.has_next_position() and elapsed_time > max_time_limit
        rospy.loginfo_once(f"Target {np.array2string(trajectory_planner.pf, precision=3)} raggiunto")
        break
    # breakFlag = False # CHECK the exit conditions
    
    log_item = {}
    q = robot.getQ() # it should contain both manipulator and base variables
    qb = q[:2]
    qm = q[2:]
    T_w_e = robot.getT_w_e().homogeneous() 
    
    # TODO: Handle the Desired trajectory    
    '''
    pos_d = trajectory_planner.next_position_sample()
    linear_vel_d = trajectory_planner.next_linear_velocity_sample()

    orient_d = trajectory_planner.next_orient_sample()
    angular_vel_d = trajectory_planner.next_angular_velocity_sample()
    '''
    
    # position error
    ep = pos_d - T_w_e[:3,3]
    # orientation error
    e_angle, e_axis = tr2angvec(orient_d @ T_w_e[:3, :3].T)
    eo = np.sin(e_angle) * e_axis

    # error norm
    e_norm = np.linalg.norm(np.concatenate((ep,eo)))

    L = L_matrix(orient_d, current_orientation)

    cmd_EE_twist = np.r_[
        linear_vel_d + Kp @ ep,
        inv(L) @ (L.T @ angular_vel_d + Ko @ eo)
    ]
    
    J = robot.whole_body_jacobian(---) # INSERT input parameters
    
    ## Weight the sub-Jacobians
    dof_base = 2
    dof_manipulator = 6
    
    W_base = np.eye(dof_base) 
    W_arm = np.eye(dof_manipulator)

    W = np.block([
        [W_base          , np.zeros((dof_base, dof_manipulator))],
        [np.zeros((dof_manipulator, dof_base)), W_arm           ]
    ])

    W_inv = np.linalg.inv(W)
    J_T = J.T

    J_pinv_weighted = W_inv @ J_T @ inv(J @ W_inv @ J_T)

    qd = clik_controller(J_pinv_weighted, err_vector)
    robot.sendQd(qd)
    
    log_item['qs'] = q.reshape((robot.model.nq,))
    log_item['dqs'] = robot.getQd().reshape((robot.model.nq,))
    # we're not saving here, but need to respect the API, 
    # hence the empty dict
    return e_norm, {}, log_item


def moveUntilContactControlLoop(args, robot : RobotManager, speed, clik_controller, i, past_data):
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
    #wrench = robot.getWrench()
    # you're already giving the speed in the EE i.e. body frame
    # so it only makes sense to have the wrench in the same frame
    #wrench = robot._getWrenchInEE()
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
    if (args.pinocchio_only) and (i > 500):
        print("let's say you hit something lule")
        breakFlag = True
    # pin.computeJointJacobian is much different than the C++ version lel
    J = pin.computeJointJacobian(robot.model, robot.data, q, robot.JOINT_ID)
    # compute the joint velocities.
    qd = clik_controller(J, speed)
    robot.sendQd(qd)
    log_item['wrench'] = wrench.reshape((6,))
    return breakFlag, {}, log_item

def moveUntilContact(args, robot, speed):
    """
    moveUntilContact
    -----
    does clik until it feels something with the f/t sensor
    """
    assert type(speed) == np.ndarray 
    clik_controller = getClikController(args)
    controlLoop = partial(moveUntilContactControlLoop, args, robot, speed, clik_controller)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {'wrench' : np.zeros(6)}
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()
    # TODO: remove, this isn't doing anything
    time.sleep(0.01)
    print("Collision detected!!")

def moveL(args, robot : RobotManager, goal_point):
    """
    moveL
    -----
    does moveL.
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    #assert type(goal_point) == pin.pinocchio_pywrap.SE3
    robot.Mgoal = copy.deepcopy(goal_point)
    clik_controller = getClikController(args)
    controlLoop = partial(controlLoopClik, robot, clik_controller)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
            'qs' : np.zeros(robot.model.nq),
            'dqs' : np.zeros(robot.model.nq),
        }
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()

# TODO: implement
def moveLFollowingLine(args, robot, goal_point):
    """
    moveLFollowingLine
    ------------------
    make a path from current to goal position, i.e.
    just a straight line between them.
    the question is what TODO with orientations.
    i suppose it makes sense to have one function that enforces/assumes
    that the start and end positions have the same orientation.
    then another version goes in a line and linearly updates the orientation
    as it goes
    """
    pass

# TODO: implement
def cartesianPathFollowing(args, robot, path):
    pass

def controlLoopCompliantClik(args, robot : RobotManager, i, past_data):
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
    save_past_dict['wrench'] = copy.deepcopy(wrench)
    wrench = args.beta * wrench + (1 - args.beta) * past_data['wrench'][-1]
    #mapping = np.zeros((6,6))
    #mapping[0:3, 0:3] = T_w_e.rotation
    #mapping[3:6, 3:6] = T_w_e.rotation
    #wrench = mapping.T @ wrench
    #Z = np.diag(np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0]))
    Z = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    wrench = Z @ wrench
    #pin.forwardKinematics(robot.model, robot.data, q)
    # first check whether we're at the goal
    SEerror = T_w_e.actInv(robot.Mgoal)
    err_vector = pin.log6(SEerror).vector 
    if np.linalg.norm(err_vector) < robot.args.goal_error:
#      print("Convergence achieved, reached destionation!")
        breakFlag = True
    # pin.computeJointJacobian is much different than the C++ version lel
    J = pin.computeJointJacobian(robot.model, robot.data, q, robot.JOINT_ID)
    # compute the joint velocities.
    # just plug and play different ones
    qd = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * args.tikhonov_damp) @ err_vector
    tau = J.T @ wrench
    #tau = tau[:6].reshape((6,1))
    qd += args.alpha * tau
    robot.sendQd(qd)
    
    log_item['qs'] = q.reshape((robot.model.nq,))
    # get actual velocity, not the commanded one.
    # although that would be fun to compare i guess
    # TODO: have both, but call the compute control signal like it should be
    log_item['dqs'] = robot.getQd().reshape((robot.model.nq,))
    log_item['wrench'] = wrench.reshape((6,))
    log_item['tau'] = tau.reshape((robot.model.nq,))
    # we're not saving here, but need to respect the API, 
    # hence the empty dict
    return breakFlag, save_past_dict, log_item

# add a threshold for the wrench
def compliantMoveL(args, robot, goal_point):
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
    clik_controller = getClikController(args)
    controlLoop = partial(controlLoopCompliantClik, args, robot)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
            'qs' : np.zeros(robot.model.nq),
            'wrench' : np.zeros(6),
            'tau' : np.zeros(robot.model.nq),
            'dqs' : np.zeros(robot.model.nq),
        }
    save_past_dict = {
            'wrench': np.zeros(6),
            }
    loop_manager = ControlLoopManager(robot, controlLoop, args, save_past_dict, log_item)
    loop_manager.run()


def clikCartesianPathIntoJointPath(args, robot, path, \
        clikController, q_init, plane_pose):
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
    # let's use the functions we already have. TODO so
    # we need to create a new RobotManager with arguments for simulation,
    # otherwise weird things will happen.
    # we keep all the other args intact
    sim_args = copy.deepcopy(args)
    sim_args.pinocchio_only = True
    sim_args.fast_simulation = True
    sim_args.real_time_plotting = False
    sim_args.visualize_manipulator = False
    sim_args.save_log = False # we're not using sim robot outside of this
    sim_args.max_iterations = 10000 # more than enough
    sim_robot = RobotManager(sim_args)
    sim_robot.q = q_init.copy()
    sim_robot._step()
    for pose in path:
        moveL(sim_args, sim_robot, pose)
        # loop logs is a dict, dict keys list preserves input order
        new_q = sim_robot.q.copy() 
        robot.updateViz(sim_robot.q, sim_robot.T_w_e)
        time.sleep(0.05)
        qs.append(new_q[:6])
        # plot this on the real robot

    ##############################################
    #  save the obtained joint-space trajectory  #
    ##############################################
    qs = np.array(qs)
    # we're putting a dmp over this so we already have the timing ready
    # TODO: make this general, you don't want to depend on other random
    # arguments (make this one traj_time, then put tau0 = traj_time there
    t = np.linspace(0, args.tau0, len(qs)).reshape((len(qs),1))
    joint_trajectory = np.hstack((t, qs))
    # TODO handle saving more consistently/intentionally
    # (although this definitely works right now and isn't bad, just mid)
    # os.makedir -p a data dir and save there, this is ugly
    np.savetxt("./joint_trajectory.csv", joint_trajectory, delimiter=',', fmt='%.5f')
    return joint_trajectory
    
    
    def whole_body_jacobian(robot : RobotManager, clik_controller, i, past_data):
        # Whole-body jacobian calculation
        I_p = np.array([1,0,0], dtype=np.float64).reshape(-1,1)
        I_o = np.array([0,0,1], dtype=np.float64).reshape(-1,1)
        O_3x1 = np.zeros((3, 1), dtype=np.float64)
    
        q = robot.getQ() # it should contain both base and manipulator/dual arm joint variables
        qb = q[:2]
        qm = q[2:] 
        Rb = rotz(qb[2]) 
            
        if robot_name == ur                
            T_w_b = robot.data.oMi[1].homogeneous() # mobile base frame wrt world frame
            T_w_e = robot.getT_w_e().homogeneous() # ee frame wrt world frame

            T_b_e = np.linalg.inv(T_w_b) @ T_w_e
            
            pos_EE_b = T_b_e[:3, 3]
            
            J_m = pin.computeFrameJacobian(robot.model, robot.data, qm, robot.ee_frame_id)

            JP_m = J_m[:3]
            JO_m = J_m[3:]

            S = (-skew(Rb@pos_EE_b)[:, 2]).reshape(-1,1) # select 3rd column

            J = np.block([
                [I_p  , S   , Rb@JP_m],
                [O_3x1, I_o ,    JO_m]
            ])
        else
            """
            FIX  
            """
            T_b_a = robot.getT_w_a().homogeneous() # transformation matrix between absolute frame and mobile base frame
            pos_a_b = T_b_a[:3, 3] # absolute frame origin

            J_m1 = pin.computeFrameJacobian(robot.model, robot.data, qm, robot.ee_frame_id) # Manipulator 1 Jacobian matrix
            J_m2 = pin.computeFrameJacobian(robot.model, robot.data, qm, robot.ee_frame_id) # Manipulator 2 Jacobian matrix

            JP_m1 = J_m1[:3]
            JO_m1 = J_m1[3:]
            JP_m2 = J_m2[:3]
            JO_m2 = J_m2[3:]

            S = (-skew(Rb@pos_a_b)[:, 2]).reshape(-1,1) # select 3rd column

            J_b = np.block([
                [I_p  , S  ],
                [O_3x1, I_o]
            ])
            
            T = 0.5 * np.block([
                [np.identity(6), np.identity(6)]
            ])

            O_3x6 = np.zeros((3, 6), dtype=np.float64)

            J_m = T @ np.block([
                    [Rb@JP_m1, O_3x6],
                    [JO_m1, O_3x6],
                    [O_3x6, Rb@ JP_m2],
                    [O_3x6, JO_m2]
                    ])

            J = np.block([[J_b,J_m]])

    return J
       
    # L-matrix calculation for angle-axis error    
    def L_matrix(orient_d, current_orientation):
    '''Calcola la matrice L, usata nel calcolo dell'errore di orientamento'''
    nd = orient_d[:, 0]  # Normal vector (x-axis)
    ad = orient_d[:, 1]  # Approach vector (y-axis)
    sd = orient_d[:, 2]  # Sliding vector (z-axis)

    ne = current_orientation[:, 0]  # Normal vector (x-axis)
    ae = current_orientation[:, 1]  # Approach vector (y-axis)
    se = current_orientation[:, 2]  # Sliding vector (z-axis)

    L = -0.5 * (skew(nd) @ skew(ne) + skew(sd) @ skew(se) + skew(ad) @ skew(ae))
    return L

    def traj(v,type):
    '''
    Calculates the time law coeffs
    The v-vector should contain:
     - "cubic": pi, vi, pf, vf, tf;
     - "quintic": pi, vi, ai, pf, vf, af, tf;
    '''
    if type == "cubic"
        t_end = v(end)
        np.array([[np.zeros((1,3)), 1],
        [np.zeros((1,2)), 1, 0],
        t_end**np.array([3:-1:0]),
        np.array([3:-1:0])* t_end ** np.array([2:-1:-1])
        ])
    # TO BE CONTINUED
        
    return L


if __name__ == "__main__": 
    args = get_args()
    robot = RobotManager(args)
    Mgoal = robot.defineGoalPointCLI()
    clik_controller = getClikController(args)
    controlLoop = partial(controlLoopClik, robot, clik_controller)
    # we're not using any past data or logging, hence the empty arguments
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, {})
    loop_manager.run()

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


def controlLoop_WholeBodyClik(goal : pin.SE3, robot : RobotManager, clik_controller, i, past_data):
    """
    controlLoop_WholeBodyClik
    ---------------
    generic control loop for clik for mobile robot
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}
    q = robot.getQ() # it should contain both manipulator and base variables
    qb = q[:3]
    qm = q[3:]
    angle = np.arctan2(q[3], q[2])
    qb[2] = angle

    T_w_e = robot.getT_w_e()
    SEerror = T_w_e.actInv(goal)
    err_vector = pin.log6(SEerror).vector 
    err_norm  = np.linalg.norm(err_vector)

    orient_d = goal.rotation

    # position error
    #ep = pos_d - T_w_e[:3,3]
    ## orientation error
    #e_angle, e_axis = tr2angvec(orient_d @ T_w_e[:3, :3].T)
    #eo = np.sin(e_angle) * e_axis

    #L = L_matrix(orient_d, current_orientation)
    ## error norm
    #e_norm = np.linalg.norm(np.concatenate((ep,eo)))

    th = 10^-3
    if e_norm > th: 
        breakFlag = True
    

    T_w_e = T_w_e.homogeneous() 
    
    # TODO: Handle the Desired trajectory    
    '''
    pos_d = trajectory_planner.next_position_sample()
    linear_vel_d = trajectory_planner.next_linear_velocity_sample()

    orient_d = trajectory_planner.next_orient_sample()
    angular_vel_d = trajectory_planner.next_angular_velocity_sample()
    '''
    

    J = whole_body_jacobian(robot) 
    
    ## Weight the sub-Jacobians
    dof_base = 3
    dof_manipulator = 6
    
    W_base = np.eye(dof_base) 
    W_arm = np.eye(dof_manipulator)

    W = np.block([
        [W_base,                                np.zeros((dof_base, dof_manipulator))],
        [np.zeros((dof_manipulator, dof_base)), W_arm                              ]])

    W_inv = np.linalg.inv(W)
    J_T = J.T

    J_pinv_weighted = W_inv @ J_T @ np.linalg.inv(J @ W_inv @ J_T)

    qd = J_pinv_weighted @ err_vector
    robot.sendQd(qd)
    
    log_item['qs'] = q.reshape((robot.model.nq,))
    log_item['dqs'] = robot.getQd().reshape((robot.model.nq,))
    # we're not saving here, but need to respect the API, 
    # hence the empty dict
    return breakFlag, {}, log_item


def whole_body_jacobian(robot : RobotManager):
    # Whole-body jacobian calculation
    I_p = np.array([1,0,0], dtype=np.float64).reshape(-1,1)
    I_o = np.array([0,0,1], dtype=np.float64).reshape(-1,1)
    O_3x1 = np.zeros((3, 1), dtype=np.float64)

    q = robot.getQ() # it should contain both base and manipulator/dual arm joint variables
    qb = q[:3]
    qm = q[3:] 
    angle = np.arctan2(q[3], q[2])
    Rb = rotz(angle) 
        
    if robot.robot_name == "heron":
        T_w_b = robot.data.oMi[1].homogeneous() # mobile base frame wrt world frame
        T_w_e = robot.getT_w_e().homogeneous() # ee frame wrt world frame

        T_b_e = np.linalg.inv(T_w_b) @ T_w_e
        
        pos_EE_b = T_b_e[:3, 3]
        
        J_m = pin.computeFrameJacobian(robot.model, robot.data, qm, robot.ee_frame_id, pin.WORLD)
        J_m = J_m[:,:-2]
        J_m = J_m[:,3:]

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

        # TODO: put left arm ee_frame_id
        J_m1 = pin.computeFrameJacobian(robot.model, robot.data, qm, robot.ee_frame_id) # Manipulator 1 Jacobian matrix
        # TODO: put right arm ee_frame_id
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
        
    return 0


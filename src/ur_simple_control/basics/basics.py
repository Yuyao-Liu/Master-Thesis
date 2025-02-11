# this is a quickest possible solution,
# not a good one
# please change at the earliest convenience

import pinocchio as pin
import numpy as np
import copy
import argparse
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager
import time
import threading
import queue

def moveJControlLoop(q_desired, robot : RobotManager, i, past_data):
    """
    moveJControlLoop
    ---------------
    most basic P control for joint space point-to-point motion, actual control loop.
    """
    breakFlag = False
    save_past_dict = {}
    # you don't even need forward kinematics for this lmao
    q = robot.getQ()
    # TODO: be more intelligent with qs
    q = q[:6]
    q_desired = q_desired[:6]
    q_error = q_desired - q

    # STOP MUCH BEFORE YOU NEED TO FOR THE DEMO
    # EVEN THIS MIGHT BE TOO MUCH
    # TODO fix later obviously
    if np.linalg.norm(q_error) < 1e-3:
        breakFlag = True
    # stupid hack, for the love of god remove this
    # but it should be small enough lel
    # there. fixed. tko radi taj i grijesi, al jebemu zivot sta je to bilo
    K = 120
    #print(q_error)
    # TODO: you should clip this
    qd = K * q_error * robot.dt
    #qd = np.clip(qd, robot.acceleration, robot.acceleration)
    robot.sendQd(qd)
    return breakFlag, {}, {}

# TODO:
# fix this by tuning or whatever else.
# MOVEL works just fine, so apply whatever's missing for there here
# and that's it.
def moveJP(args, robot, q_desired):
    """
    moveJP
    ---------------
    most basic P control for joint space point-to-point motion.
    just starts the control loop without any logging.
    """
    assert type(q_desired) == np.ndarray
    controlLoop = partial(moveJControlLoop, q_desired, robot)
    # we're not using any past data or logging, hence the empty arguments
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, {})
    loop_manager.run()
    # TODO: remove, this isn't doing anything
    #time.sleep(0.01)
    if args.debug_prints:
        print("MoveJP done: convergence achieved, reached destionation!")


# NOTE: it's probably a good idea to generalize this for different references:
# - only qs
# - qs and vs
# - whatever special case
# it could be better to have separate functions, whatever makes the code as readable 
# as possible.
# also NOTE: there is an argument to be made for pre-interpolating the reference.
# this is because joint positions will be more accurate.
# if we integrate them with interpolated velocities.
def followKinematicJointTrajPControlLoop(stop_at_final : bool, robot: RobotManager, reference, i, past_data):
    breakFlag = False
    save_past_dict = {}
    log_item = {}
    q = robot.getQ()
    v = robot.getQd()
    # NOTE: assuming we haven't missed a timestep,
    # which is pretty much true
    t = i * robot.dt
    # take the future (next) one
    t_index_lower = int(np.floor(t / reference['dt']))
    t_index_upper = int(np.ceil(t / reference['dt']))

    # TODO: set q_refs and v_refs once (merge with interpolation if)
    if t_index_upper >= len(reference['qs']) - 1:
        t_index_upper = len(reference['qs']) - 1
    q_ref = reference['qs'][t_index_upper]
    if (t_index_upper == len(reference['qs']) - 1) and stop_at_final:
        v_ref = np.zeros(robot.model.nv)
    else:
        v_ref = reference['vs'][t_index_upper]
    
    # TODO NOTE TODO TODO: move under stop/don't stop at final argument
    if (t_index_upper == len(reference['qs']) - 1) and \
            (np.linalg.norm(q - reference['qs'][-1]) < 1e-2) and \
            (np.linalg.norm(v) < 1e-2):
        breakFlag = True

    # TODO: move interpolation to a different function later
    if (t_index_upper < len(reference['qs']) - 1) and (not breakFlag):
        #angle = (reference['qs'][t_index_upper] - reference['qs'][t_index_lower]) / reference['dt']
        angle_v = (reference['vs'][t_index_upper] - reference['vs'][t_index_lower]) / reference['dt']
        time_difference =t - t_index_lower * reference['dt']
        v_ref = reference['vs'][t_index_lower] + angle_v * time_difference
        # using pin.integrate to make this work for all joint types
        # NOTE: not fully accurate as it could have been integrated with previous interpolated velocities,
        # but let's go for it as-is for now
        # TODO: make that work via past_data IF this is still too saw-looking
        q_ref = pin.integrate(robot.model, reference['qs'][t_index_lower], reference['vs'][t_index_lower] * time_difference)


    # TODO: why not use pin.difference for both?
    if robot.robot_name == "ur5e":
        error_q = q_ref - q
    if robot.robot_name == "heron":
        error_q = pin.difference(robot.model, q, q_ref) #/ robot.dt
    error_v = v_ref - v
    Kp = 1.0
    Kd = 0.5

    #          feedforward                      feedback 
    v_cmd = v_ref + Kp * error_q #+ Kd * error_v
    #qd_cmd = v_cmd[:6]
    robot.sendQd(v_cmd)

    log_item['error_qs'] = error_q
    log_item['error_vs'] = error_v
    log_item['qs'] = q
    log_item['vs'] = v
    log_item['vs_cmd'] = v_cmd
    log_item['reference_qs'] = q_ref
    log_item['reference_vs'] = v_ref

    return breakFlag, {}, log_item

def followKinematicJointTrajP(args, robot, reference):
    # we're not using any past data or logging, hence the empty arguments
    controlLoop = partial(followKinematicJointTrajPControlLoop, args.stop_at_final, robot, reference)
    log_item = {
        'error_qs' : np.zeros(robot.model.nq),
        'error_vs' : np.zeros(robot.model.nv),
        'qs' : np.zeros(robot.model.nq),
        'vs' : np.zeros(robot.model.nv),
        'vs_cmd' : np.zeros(robot.model.nv),
        'reference_qs' : np.zeros(robot.model.nq),
        'reference_vs' : np.zeros(robot.model.nv)
    }
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()
    if args.debug_prints:
        print("followKinematicJointTrajP done: reached path destionation!")
    


def moveJPIControlLoop(q_desired, robot : RobotManager, i, past_data):
    """
    PID control for joint space point-to-point motion with approximated joint velocities.
    """

    # ================================
    # Initialization
    # ================================
    breakFlag = False
    save_past_dict = {}
    log_item = {}

    # ================================
    # Current Joint Positions
    # ================================
    q = robot.getQ()[:6]  # Get current joint positions (first 6 joints)

    # ================================
    # Retrieve Previous States
    # ================================
    q_prev = past_data['q_prev'][-1]    # Previous joint positions
    e_prev = past_data['e_prev'][-1]    # Previous position error

    qd_actual = robot.getQd()[:6]

    # ================================
    # Compute Position Error
    # ================================
    q_error = q_desired - q  # Position error


    # ================================
    # Check for Convergence
    # ================================
    if np.linalg.norm(q_error) < 1e-3 and np.linalg.norm(qd_actual) < 1e-3:
        breakFlag = True  # Convergence achieved

    # ================================
    # Update Integral of Error
    # ================================
    integral_error = past_data['integral_error'][-1]
    integral_error = np.array(integral_error, dtype=np.float64).flatten()
    integral_error += q_error * robot.dt  # Accumulate error over time

    # Anti-windup: Limit integral error to prevent excessive accumulation
    max_integral = 10
    integral_error = np.clip(integral_error, -max_integral, max_integral)

    # ================================
    # Save Current States for Next Iteration
    # ================================
    save_past_dict['integral_error'] = integral_error  # Save updated integral error
    save_past_dict['q_prev'] = q                       # Save current joint positions
    save_past_dict['e_prev'] = q_error                 # Save current position error

    # ================================
    # Control Gains
    # ================================
    Kp = 7.0  # Proportional gain
    Ki = 0.0  # Integral gain

    # ================================
    # Compute Control Input (Joint Velocities)
    # ================================
    qd = Kp * q_error + Ki * integral_error 
    #qd[5]=qd[5]*10

    # ================================
    # Send Joint Velocities to the Robot
    # ================================
    robot.sendQd(qd)

    qd = robot.getQd()
    log_item['qs'] = q
    log_item['dqs'] = qd[:6]
    log_item['integral_error'] = integral_error  # Save updated integral error
    log_item['e_prev'] = q_error                 # Save current position error
    return breakFlag, save_past_dict, log_item


def moveJPI(args, robot, q_desired):
    assert isinstance(q_desired, np.ndarray)
    controlLoop = partial(moveJPIControlLoop, q_desired, robot)

    # ================================
    # Initialization
    # ================================
    # Get initial joint positions
    initial_q = robot.getQ()[:6]

    # Initialize past data for control loop
    save_past_dict = {
        'integral_error': np.zeros(robot.model.nq)[:6],
        'q_prev':         initial_q,
        'e_prev':         q_desired - initial_q,   # Initial position error (may need to be q_desired - initial_q)
    }

    # Initialize log item (if logging is needed)
    log_item = {
        'qs'            : np.zeros(6),
        'dqs'           : np.zeros(6),
        'integral_error': np.zeros(robot.model.nq)[:6],
        'e_prev'        : q_desired - initial_q,   # Initial position error (may need to be q_desired - initial_q)
        }  

    # ================================
    # Create and Run Control Loop Manager
    # ================================
    loop_manager = ControlLoopManager(robot, controlLoop, args, save_past_dict, log_item)
    loop_manager.run()

    # ================================
    # Debug Printing
    # ================================
    if args.debug_prints:
        print("MoveJPI done: convergence achieved, reached destination!")


def freedriveControlLoop(args, robot : RobotManager, com_queue, pose_n_q_dict, i, past_data):
    """
    controlLoopFreedrive
    -----------------------------
    while in freedrive, collect qs.
    this can be used to visualize and plot while in freedrive,
    collect some points or a trajectory etc.
    this function does not have those features,
    but you can use this function as template to make them
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.getQ()
    wrench = robot.getWrench()
    T_w_e = robot.getT_w_e()

    if not com_queue.empty():
        msg = com_queue.get()
        if msg == 'q':
            breakFlag = True
        if msg == 's':
            pose_n_q_dict['T_w_es'].append(T_w_e.copy())
            pose_n_q_dict['qs'].append(q.copy())

    if args.debug_prints:
        print("===========================================")
        print(T_w_e)
        print("q:", *np.array(q).round(4))

    log_item['qs'] = q.reshape((robot.model.nq,))
    log_item['dqs'] = robot.getQd().reshape((robot.model.nv,))
    return breakFlag, save_past_dict, log_item

def freedriveUntilKeyboard(args, robot : RobotManager):
    """
    controlLoopFreedrive
    -----------------------------
    while in freedrive, collect qs.
    this can be used to visualize and plot while in freedrive,
    collect some points or a trajectory etc.
    you can save the log from this function and use that,
    or type on the keyboard to save specific points (q + T_w_e)
    """
    if args.pinocchio_only:
        print("""
    ideally now you would use some sliders or something, 
    but i don't have time to implement that. just run some movement 
    to get it where you want pls. freedrive will just exit now
            """)
        return {}
    robot.setFreedrive()
    # set up freedrive controlloop (does nothing, just accesses
    # all controlLoopManager goodies)
    log_item = {'qs'  : np.zeros((robot.model.nq,)),
                'dqs' : np.zeros((robot.model.nv,))
                }
    save_past_dict = {}
    # use queue.queue because we're doing this on a
    # threading level, not multiprocess level
    # infinite size (won't need more than 1 but who cares)
    com_queue = queue.Queue()
    # we're passing pose_n_q_list by reference
    # (python default for mutables) 
    pose_n_q_dict = {'T_w_es' : [], 'qs': []}
    controlLoop = ControlLoopManager(robot, partial(freedriveControlLoop, args, robot, com_queue, pose_n_q_dict), args, save_past_dict, log_item) 

    # wait for keyboard input in a different thread
    # (obviously necessary because otherwise literally nothing else 
    #  can happen)
    def waitKeyboardFunction(com_queue):
        cmd = ""
        # empty string is cast to false
        while True:
            cmd = input("Press q to stop and exit, s to save joint angle and T_w_e: ")
            if (cmd != "q") and (cmd != "s"):
                print("invalid input, only s or q (then Enter) allowed")
            else:
                com_queue.put(cmd)
                if cmd == "q":
                    break

    # we definitely want a thread and leverage GIL,
    # because the thread is literally just something to sit
    # on a blocking call from keyboard input
    # (it would almost certainly be the same without the GIL,
    #  but would maybe require stdin sharing or something)
    waitKeyboardThread = threading.Thread(target=waitKeyboardFunction, args=(com_queue,))
    waitKeyboardThread.start()
    controlLoop.run()
    waitKeyboardThread.join()
    robot.unSetFreedrive()
    return pose_n_q_dict

if __name__ == "__main__":
    parser = getMinimalArgParser()
    args = parser.parse_args()
    robot = RobotManager(args)
    robot._step()
    print(robot.q)
    q_goal = np.random.random(6) * 2*np.pi - np.pi
    print(q_goal)
    #moveJPI(args, robot, q_goal)
    moveJP(args, robot, q_goal)
    robot.killManipulatorVisualizer()

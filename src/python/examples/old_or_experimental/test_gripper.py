# just import everything
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import time
from functools import partial
from ur_simple_control.visualize.visualize import plotFromDict
from ur_simple_control.util.draw_path import drawPath
from ur_simple_control.managers import ControlLoopManager, RobotManager
# needed to access some functions directly,
# TODO: hide this for later use obviously
from ur_simple_control.util.robotiq_gripper import RobotiqGripper

# TODO have a generic get_args function 
# that just has all of them somewhere in utils
# so that i don't have to copy-paste this around
# for simple stuff like this gripper test
def get_args():
    parser = argparse.ArgumentParser(description='Run closed loop inverse kinematics \
            of various kinds. Make sure you know what the goal is before you run!',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--simulation', action=argparse.BooleanOptionalAction, 
            help="whether you are running the UR simulator", default=False)
    parser.add_argument('--pinocchio-only', action=argparse.BooleanOptionalAction, 
            help="whether you want to just integrate with pinocchio", default=False)
    parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, 
            help="whether you want to visualize with gepetto, but NOTE: not implemented yet", default=False)
    parser.add_argument('--gripper', action=argparse.BooleanOptionalAction, \
            help="whether you're using the gripper", default=True)
    parser.add_argument('--goal-error', type=float, \
            help="the final position error you are happy with", default=1e-2)
    parser.add_argument('--max-iterations', type=int, \
            help="maximum allowable iteration number (it runs at 500Hz)", default=100000)
    parser.add_argument('--acceleration', type=float, \
            help="robot's joints acceleration. scalar positive constant, max 1.7, and default 0.4. \
                   BE CAREFUL WITH THIS. the urscript doc says this is 'lead axis acceleration'.\
                   TODO: check what this means", default=0.3)
    parser.add_argument('--speed-slider', type=float,\
            help="cap robot's speed with the speed slider \
                    to something between 0 and 1, 0.5 by default \
                    BE CAREFUL WITH THIS.", default=0.5)
    parser.add_argument('--tikhonov-damp', type=float, \
            help="damping scalar in tikhonov regularization", default=1e-3)
    # TODO add the rest
    parser.add_argument('--clik-controller', type=str, \
            help="select which click algorithm you want", \
            default='dampedPseudoinverse', choices=['dampedPseudoinverse', 'jacobianTranspose'])
        # maybe you want to scale the control signal
    parser.add_argument('--controller-speed-scaling', type=float, \
            default='1.0', help='not actually_used atm')

    args = parser.parse_args()
    if args.gripper and args.simulation:
        raise NotImplementedError('Did not figure out how to put the gripper in \
                the simulation yet, sorry :/ . You can have only 1 these flags right now')
    return args

def readFromGripper(robot):
    did = 0
    robot.gripper.move_and_wait_for_pos(0,255,1)
    time.sleep(2)
    robot.gripper.move_and_wait_for_pos(255,255,1)
    time.sleep(2)
    # real value
    #offset_open = 218
    offset_open = 255

    for i in range(5):
        robot.gripper.move_and_wait_for_pos(offset_open, 1, 1)
        print("dropping", offset_open)
        print("current_position", robot.gripper.get_current_position())
        time.sleep(4)
        
#        exit() 

        robot.gripper.move_and_wait_for_pos(255, 1, 255)
        print("holding", 255)
        print("current_position", robot.gripper.get_current_position())
        did += 1
        time.sleep(5)
#    for i in range(255):
#        start = time.time()
#        robot.gripper.move_and_wait_for_pos(offset - i, 255, 10)
#        print(offset - i, robot.gripper.get_current_position())
#        did += 1
#        end = time.time()
#        #time.sleep(10/125 - (end - start))
#        time.sleep(5)
#    print("read", did, "times")
#    robot.gripper.move(255, 255, 255)
#    for i in range(1250):
#        print(i, robot.gripper.get_current_position())
#        did += 1
#        time.sleep(1/125)


# TODO: when this makes some sense,
# integrate it into the RobotManager class
# let's call the gripper's position x for now
# then velocity is xd and acceleration is xdd
# TODO ALSO a problem:
# you're supposed to send this at 125Hz, not 500!
# but let's first try 500 and see what happens.
# if there will be problems, implement handling that into the 
# controlLoop manager. count some seconds there, or have a different 
# thread for the gripper (will be painfull, try getting
# away with 100Hz since that's a multiple of 500, no need 
# to overengineer if it isn't necessary)
# TODO: add control frequency as a parameter to controlLoop
def gripperPositionControlLoop(robot, clik_controller, i, past_data):
    breakFlag = False
    log_item = {}
    
    # let's start simple:
    # send position command, read as it goes (essential if you want 
    # to have a closed-loop system)
    x_goal = robot.gripper.get_min_position() # or whatever really
    # speed is in [0,255]
    xd = 10
    # force is in [0,255] - acts while gripping I guess
    F_x= 10
    # you'll need to pass some kind of a flag here
    robot.gripper.move(x_goal, xd, F_x)
    # wait until not moving
    cur_obj = self._get_var(self.OBJ)
    # TODO: this can't actually be in this control loop form
    # so TODO: make it separate function for now, then a controlLoop?
    # also NOTE: this is code straight from the RobotiqGripper class
    while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
        cur_obj = robot.gripper._get_var(self.OBJ)
        print(cur_obj)

#    log_item['x'] = x.reshape((,))
#    log_item['xd'] = xd.reshape((,))
#    log_item['xdd'] = xdd.reshape((,))
    return breakFlag, {}, log_item


# TODO: implement via hacks
def gripperLinearSlipControlLoop(robot, clik_controller, i, past_data):
    breakFlag = False
    log_item = {}
    
    # let's start simple:
    # send position command, read as it goes (essential if you want 
    # to have a closed-loop system)
    x_goal = robot.gripper.get_min_position() # or whatever really
    # speed is in [0,255]
    xd = 10
    # force is in [0,255] - acts while gripping I guess
    F_x= 10
    # you'll need to pass some kind of a flag here
    robot.gripper.move(x_goal, xd, F_x)
    # wait until not moving
    cur_obj = self._get_var(self.OBJ)
    # TODO: this can't actually be in this control loop form
    # so TODO: make it separate function for now, then a controlLoop?
    # also NOTE: this is code straight from the RobotiqGripper class
    while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
        cur_obj = robot.gripper._get_var(self.OBJ)
        print(cur_obj)

#    log_item['x'] = x.reshape((,))
#    log_item['xd'] = xd.reshape((,))
#    log_item['xdd'] = xdd.reshape((,))
    return breakFlag, {}, log_item

# TODO: implement as per yiannis' paper
def gripperTorsionalSlipControlLoop(robot, clik_controller, i, past_data):
    breakFlag = False
    log_item = {}
    
    # let's start simple:
    # send position command, read as it goes (essential if you want 
    # to have a closed-loop system)
    x_goal = robot.gripper.get_min_position() # or whatever really
    # speed is in [0,255]
    xd = 10
    # force is in [0,255] - acts while gripping I guess
    F_x= 10
    # you'll need to pass some kind of a flag here
    robot.gripper.move(x_goal, xd, F_x)
    # wait until not moving
    cur_obj = self._get_var(self.OBJ)
    # TODO: this can't actually be in this control loop form
    # so TODO: make it separate function for now, then a controlLoop?
    # also NOTE: this is code straight from the RobotiqGripper class
    while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
        cur_obj = robot.gripper._get_var(self.OBJ)
        print(cur_obj)

#    log_item['x'] = x.reshape((,))
#    log_item['xd'] = xd.reshape((,))
#    log_item['xdd'] = xdd.reshape((,))
    return breakFlag, {}, log_item


if __name__ == "__main__": 
    args = get_args()
    robot = RobotManager(args)
    readFromGripper(robot)

    # real stuff for later
#    controlLoop = partial(gripperPositionControlLoop, robot, clik_controller)
#    log_dict = {
#            'x' : np.zeros((args.max_iterations, 1)),
#            'xd' : np.zeros((args.max_iterations, 1)),
#            'xdd' : np.zeros((args.max_iterations, 1)),
#        }
#    # we're not using any past data or logging, hence the empty arguments
#    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_dict)
#    loop_manager.run()
#    saveLog(log_dict, final_iteration, args)


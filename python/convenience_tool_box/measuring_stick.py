import pinocchio as pin
import numpy as np
import sys
import os
from os.path import dirname, join, abspath
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import os
import copy
import signal
from ur_simple_control.managers import RobotManager
from ur_simple_control.clik.clik import getClikArgs

def handler(signum, frame):
    print('i will end freedrive and exit')
    robot.rtde_control.endFreedriveMode()
    exit()

args = get_args()
robot = RobotManager(args)


robot.rtde_control.freedriveMode()
signal.signal(signal.SIGINT, handler)


while True:
    Mtool = robot.getT_w_e() 
    print(Mtool)
    print("pin:", *Mtool.translation.round(4), *pin.rpy.matrixToRpy(Mtool.rotation).round(4))
    print("ur5:", *np.array(robot.rtde_receive.getActualTCPPose()).round(4))
    time.sleep(0.005)

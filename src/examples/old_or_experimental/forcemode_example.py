from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import pinocchio as pin
import numpy as np
import os
import time
import signal
import matplotlib.pyplot as plt

rtde_control = RTDEControlInterface("192.168.1.102")
rtde_receive = RTDEReceiveInterface("192.168.1.102")

def handler(signum, frame):
    rtde_control.forceModeStop()
    print('sending 100 speedjs full of zeros')
    for i in range(100):
        vel_cmd = np.zeros(6)
        rtde_control.speedJ(vel_cmd, 0.1, 1.0 / 500)
    rtde_control.stopScript()
    exit()

signal.signal(signal.SIGINT, handler)

# task frame defines force frame relative to base frame
task_frame = [0, 0, 0, 0, 0, 0]
# these are in {0,1} and select which task frame direction compliance is active in
selection_vector = [1, 1, 0, 0, 0, 0]
# the wrench applied to the environment: 
# position is adjusted to achieve the specified wrench
wrench = [-10, 0, 0, 0, 0, 0]
# type is in {1,2,3}
# 1: force frame is transformed so that it's y-axis is aligned
#    with a vector from tcp to origin of force frame (what and why??)
# 2: force frame is not transformed
# 3: transforms force frame s.t. it's x-axis is the projection of
#    tcp velocity to the x-y plane of the force frame (again, what and why??)
ftype = 2
# limits for:
# - compliant axes: highest tcp velocities allowed on compliant axes
# - non-compliant axes: maximum tcp position error compared to the program (which prg,
#                       and where is this set?)
limits = [2, 2, 1.5, 1, 1, 1]
# NOTE: avoid movements parallel to compliant direction
# NOTE: avoid high a-/de-celeration because this decreses force control accuracy
# there's also force_mode_set_damping:
# - A value of 1 is full damping, so the robot will decellerate quickly if no
#   force is present. A value of 0 is no damping, here the robot will maintain
#   the speed. 
# - call this before calling force mode if you want it to work.

update_rate = 500
dt = 1 / update_rate

q = np.array(rtde_receive.getActualQ())
rtde_control.forceMode(task_frame, selection_vector, wrench, ftype, limits)

for i in range(20000):
    start = time.time()
    if i > 10000:
        wrench[0] = 10
        rtde_control.forceMode(task_frame, selection_vector, wrench, ftype, limits)
    else:
        rtde_control.forceMode(task_frame, selection_vector, wrench, ftype, limits)
    end = time.time()
    diff = end - start
    if dt < diff:
        print("missed deadline by", diff - dt)
        continue
    else:
        time.sleep(dt - diff)

rtde_control.forceModeStop()
rtde_control.stopScript()

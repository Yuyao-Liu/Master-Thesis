# idc if i need this
# Implement the default Matplotlib key bindings.
from robot_stuff.InverseKinematics import InverseKinematicsEnv
from robot_stuff.drawing import *
from robot_stuff.inv_kinm import *

import numpy as np
import matplotlib.pyplot as plt
import time

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

#rtde_c = RTDEControl("192.168.1.102")
rtde_c = RTDEControl("127.0.0.1")

#rtde_r = RTDEReceive("192.168.1.102")
rtde_r = RTDEReceive("127.0.0.1")
q = rtde_r.getActualQ()

# Parameters
acceleration = 0.5
dt = 1.0/500  # 2ms

ik_env = InverseKinematicsEnv()
ik_env.damping = 25
ik_env.goal[0] = -0.21
ik_env.goal[1] = -0.38
ik_env.goal[2] = 0.2
ik_env.render()


ik_env.robots[0].setJoints(q)
ik_env.robots[0].calcJacobian()
print(ik_env.robots[0].p_e)
#print(type(init_q))
#exit()

# putting it into this class so that python remembers it 'cos reasons, whatever
controller = invKinm_dampedSquares
#controller = invKinmSingAvoidanceWithQP_kI
#while True:
while True:
    start = time.time()
    q = rtde_r.getActualQ()
    ik_env.robots[0].setJoints(q)
    ik_env.robots[0].calcJacobian()
    q_dots = controller(ik_env.robots[0], ik_env.goal)
    thetas = np.array([joint.theta for joint in ik_env.robots[0].joints])
    ik_env.render()
    #print(ik_env.robot.p_e.copy())
    #print(ik_env.goal)
    distance = np.linalg.norm(ik_env.robots[0].p_e.copy() - ik_env.goal)
    print(distance)
    if distance < 0.01:
        t_start = rtde_c.initPeriod()
        t_start = rtde_c.initPeriod()
        rtde_c.speedJ(np.zeros(6), acceleration, dt)
        rtde_c.waitPeriod(t_start)
        break

#    print(q_dots)
#    print(thetas)
    
    end = time.time()
    print("time on rendering", end - start)
    print("fps: ", 1/ (end - start))
    t_start = rtde_c.initPeriod()
    rtde_c.speedJ(q_dots, acceleration, dt)
    rtde_c.waitPeriod(t_start)

for i in range(20):
    t_start = rtde_c.initPeriod()
    rtde_c.speedJ(np.zeros(6), acceleration, dt)
    rtde_c.waitPeriod(t_start)


print("done")

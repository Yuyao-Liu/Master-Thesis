import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import signal
from ur_simple_control.managers import RobotManager
from ur_simple_control.clik.clik_point_to_point import get_args

args = get_args()
robot = RobotManager(args)

#print("payload", robot.rtde_receive.getPayload())
#print("payload cog", robot.rtde_receive.getPayloadCog())
#print("payload ixx, iyy, izz, angular", robot.rtde_receive.getPayloadInertia())

ft_readings = []
dt = 1/500
#while True:
for i in range(5000):
    start = time.time()
    q = robot.rtde_receive.getActualQ()
    ft = robot.rtde_receive.getActualTCPForce()
    tau = robot.rtde_control.getJointTorques()
    current = robot.rtde_receive.getActualCurrent()
    q.append(0.0)
    q.append(0.0)
    pinMtool = robot.getT_w_e()
    
    if i % 25 == 0:
        print("ur5:", *np.array(robot.rtde_receive.getActualTCPPose()).round(4))
        print("pin:", *pinMtool.translation.round(4), *pin.rpy.matrixToRpy(pinMtool.rotation).round(4))
    #print("current", current)
    #print("getActualTCPForce", ft)
    #print("tau", tau)
    ft_readings.append(ft)
    end = time.time()
    diff = end - start
    if diff < dt:
        time.sleep(dt - diff)


ft_readings = np.array(ft_readings)
time = np.arange(len(ft_readings))
plt.title('fts')
ax = plt.subplot(231)
ax.plot(time, ft_readings[:,0])
ax = plt.subplot(232)
ax.plot(time, ft_readings[:,1])
ax = plt.subplot(233)
ax.plot(time, ft_readings[:,2])
ax = plt.subplot(234)
ax.plot(time, ft_readings[:,3])
ax = plt.subplot(235)
ax.plot(time, ft_readings[:,4])
ax = plt.subplot(236)
ax.plot(time, ft_readings[:,5])
print("average over time", np.average(ft_readings, axis=0))
plt.savefig('fts.png', dpi=600)
plt.show()
#    ft = rtde_receive.getFtRawWrench()
#    print("getFtRawWrench", ft)
#    print("payload inertia", rtde_receive.getPayloadInertia())
#    print("momentum", rtde_receive.getActualMomentum())
#    print("target qdd", rtde_receive.getTargetQdd())
#    print("robot_current", rtde_receive.getActualRobotCurrent())
#    print("joint_voltage", rtde_receive.getActualJointVoltage())
#    print("robot_voltage", rtde_receive.getActualRobotVoltage())
#    print("getSafetyMode", rtde_receive.getSafetyMode())
#    print("tool_accelerometer", rtde_receive.getActualToolAccelerometer())
#    q.append(0.0)
#    q.append(0.0)
#    pin.forwardKinematics(model, data, np.array(q))
#    print(data.oMi[6])
#    print("pin:", *data.oMi[6].translation.round(4), *pin.rpy.matrixToRpy(data.oMi[6].rotation).round(4))
#    print("ur5:", *np.array(rtde_receive.getActualTCPPose()).round(4))
#    time.sleep(0.005)

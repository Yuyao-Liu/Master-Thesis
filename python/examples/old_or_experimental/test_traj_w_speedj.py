from create_dmp import DMP
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from robotiq_gripper import RobotiqGripper
import pinocchio as pin
import numpy as np
import os
import time
import signal
import matplotlib.pyplot as plt


#urdf_path_relative = "../../robot_descriptions/urdf/ur5e_with_robotiq_hande.urdf"
#urdf_path_absolute = os.path.abspath(urdf_path_relative)
#mesh_dir = "../../robot_descriptions/"
#mesh_dir_absolute = os.path.abspath(mesh_dir)
##print(mesh_dir_absolute)
#model = pin.buildModelFromUrdf(urdf_path_absolute)
##print(model)
#data = pin.Data(model)
##print(data)

rtde_control = RTDEControl("192.168.1.102")
rtde_receive = RTDEReceive("192.168.1.102")

trajectory_loadpath = './ur10_omega_trajectory.csv'
data = np.genfromtxt(trajectory_loadpath, delimiter=',')
t = data[:, 0]
t = t.reshape(1, len(t))
y = np.array(data[:, 1:]).T

update_rate = 500
dt = 1.0 / update_rate
kp = 2

n = y.shape[0]
yd = (y[:, 1:] - y[:, :-1]) / (t[0, 1:] - t[0, :-1])
yd = np.concatenate((yd, np.zeros((n, 1))), axis=1)
print(yd.shape)
rtde_control.moveJ(y[:,0])
for i in range(yd.shape[1]):
    start = time.time()
    q = np.array(rtde_receive.getActualQ())
    vel_cmd = yd[:,i] + kp * (y[:,i] - q.reshape((6,1)))
    rtde_control.speedJ(yd[:,i], 0.1, 1/500)
    end = time.time()
    diff = end - start
    if dt < diff:
        continue
    else:
        time.sleep(dt - diff)

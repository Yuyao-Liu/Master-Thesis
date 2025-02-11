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
time = data[:, 0]
time = time.reshape(1, len(time))
y = np.array(data[:, 1:])

for i in range(len(y)):
    if i % 100 == 0:
        rtde_control.moveJ(y[i])


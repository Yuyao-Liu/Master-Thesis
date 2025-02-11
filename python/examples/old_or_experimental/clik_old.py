import pinocchio as pin
import numpy as np
import sys
import os
from os.path import dirname, join, abspath
import time
from pinocchio.visualize import GepettoVisualizer
#import gepetto.corbaserver
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface
from ur_simple_control.util.robotiq_gripper import RobotiqGripper
import os
import copy
import signal
sys.path.insert(0, '../util')
from ur_simple_control.util.get_model import get_model

#SIMULATION = True
SIMULATION = False
PINOCCHIO_ONLY = False
# urdf treats gripper as two prismatic joints, but they do not affect the overall movement
# of the robot, so we cut jacobian calculations at the 6th joint (which also doesn't move position-wise,
# but it does change the orientation)

# run visualizer from code (not working atm)
#gepetto.corbaserver.start_server()
#time.sleep(3)

# load model
VISUALIZE = False
#urdf_path_relative = "../robot_descriptions/urdf/ur5e_with_robotiq_hande.urdf"
#urdf_path_absolute = os.path.abspath(urdf_path_relative)
#mesh_dir = "../robot_descriptions/"
#mesh_dir_absolute = os.path.abspath(mesh_dir)
model, collison_model, visual_mode, data = get_model(VISUALIZE)


#load gripper
gripper = RobotiqGripper()

#viz.display(q0)


if not SIMULATION: 
    rtde_control = RTDEControl("192.168.1.102")
    rtde_receive = RTDEReceive("192.168.1.102")
    rtde_io = RTDEIOInterface("192.168.1.102")
    #NOTE: socket_timeout is the third argument, check what it does 
    gripper.connect("192.168.1.102", 63352)
    # this is a blocking call
    gripper.activate()
else:
    rtde_control = RTDEControl("127.0.0.1")
    rtde_receive = RTDEReceive("127.0.0.1")
    # this can't work until you put the gripper in the simulation
    #gripper.connect("127.0.0.1", 63352)

# define goal

JOINT_ID = 6
q = rtde_receive.getActualQ()
q.append(0.0)
q.append(0.0)
pin.forwardKinematics(model, data, np.array(q))
Mtool = data.oMi[JOINT_ID]
print("pos", Mtool)
#SEerror = pin.SE3(np.zeros((3,3)), np.array([0.0, 0.0, 0.1])
Mgoal = copy.deepcopy(Mtool)
Mgoal.translation = Mgoal.translation + np.array([0.0, 0.0, 0.1])
print("goal", Mgoal)
eps = 1e-3
IT_MAX = 100000
update_rate = 500
dt = 1/update_rate
damp = 1e-6
# nice but large
#acceleration = 1.0
acceleration = 0.2
# let's go nice and slow
rtde_io.setSpeedSlider(0.5)

# if you just stop it normally, it will continue running
# the last speedj lmao
# there's also an actual stop command
def handler(signum, frame):
    print('sending 100 speedjs full of zeros')
    for i in range(100):
        vel_cmd = np.zeros(6)
        rtde_control.speedJ(vel_cmd, 0.1, 1.0 / 500)
    exit()

signal.signal(signal.SIGINT, handler)

success = False
for i in range(IT_MAX): 
    start = time.time()
    q = rtde_receive.getActualQ()
    if not SIMULATION:
        gripper_pos = gripper.get_current_position()
        # all 3 are between 0 and 255
        #gripper.move(i % 255, 100, 100)
        # just padding to fill q, which is only needed for forward kinematics
        #q.append(gripper_pos)
        #q.append(gripper_pos)
        q.append(0.0)
        q.append(0.0)
        # pinocchio wants an ndarray
        q = np.array(q)
    pin.forwardKinematics(model, data, q)
    SEerror = data.oMi[JOINT_ID].actInv(Mgoal)
    err_vector = pin.log6(SEerror).vector 
    if np.linalg.norm(err_vector) < eps:
      success = True
      print("reached destionation")
      break
    if i >= IT_MAX: 
        success = false
        print("FAIL: did not succed in IT_MAX iterations")
        break
    # this does the whole thing unlike the C++ version lel
    J = pin.computeJointJacobian(model, data, q, JOINT_ID)
    #J = J + np.eye(J.shape[0], J.shape[1]) * 10**-4
    # idk what i was thinking here lol
    #v = np.matmul(np.linalg.pinv(J), pin.log(SEerror.inverse() * Mgoal).vector)
    #v = J.T @ err_vector
    #v = np.linalg.pinv(J) @ err_vector
    v = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0], J.shape[0]) * 10**-2) @ err_vector
    v_cmd = v[:6]
    v_cmd = np.clip(v_cmd, -2, 2)
    if not SIMULATION:
        rtde_control.speedJ(v_cmd, acceleration, dt)
    else:
        q = pin.integrate(model, q, v * dt)
    if not i % 1000:
        print("pos", data.oMi[JOINT_ID])
        print("linear error = ", pin.log6(SEerror).linear)
        print("angular error = ", pin.log6(SEerror).angular)
        print(" error = ", err_vector.transpose())
    end = time.time()
    diff = end - start
    if dt < diff:
        print("missed deadline by", diff - dt)
        continue
    else:
        time.sleep(dt - diff)

if success: 
    print("Convergence achieved!")
else:
    print("Warning: the iterative algorithm has not reached convergence to the desired precision")

print("final error", err_vector.transpose()) 
handler(None, None)

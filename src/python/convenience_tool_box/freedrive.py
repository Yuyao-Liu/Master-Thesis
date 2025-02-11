import numpy as np
import time
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import os
import copy
import signal

def handler(signum, frame):
    print('i will end freedrive and exit')
    rtde_control.endFreedriveMode()
    exit()


rtde_control = RTDEControlInterface("192.168.1.103")
rtde_receive = RTDEReceiveInterface("192.168.1.103")
rtde_io = RTDEIOInterface("192.168.1.103")
rtde_io.setSpeedSlider(0.2)
while not rtde_control.isConnected():
    continue
print("connected")

rtde_control.freedriveMode()
signal.signal(signal.SIGINT, handler)

while True:
    q = rtde_receive.getActualQ()
    q = np.array(q)
    print(*q.round(3), sep=', ')
    time.sleep(0.005)

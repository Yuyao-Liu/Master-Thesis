import pinocchio as pin
import numpy as np
from ur_simple_control.managers import (
    ProcessManager,
    getMinimalArgParser,
)
from ur_simple_control.networking.server import server
from ur_simple_control.networking.client import client
import time


def getArgs():
    parser = getMinimalArgParser()
    parser.description = "simple test program. sends commands to go in a circle, receives and prints out wrenches received by the client executing the go-in-circle commands"
    parser.add_argument("--host", type=str, help="host ip address", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="host's port", default=7777)

    args = parser.parse_args()
    return args


def sendCircleGoalReceiveWrench(freq, i):
    save_past_dict = {}
    log_item = {}
    breakFlag = False

    # create command to go in a circle
    radius = 0.6
    t = (i / freq) / 6
    pose = pin.SE3.Identity()
    pose.translation = radius * np.array([np.cos(t), np.sin(t), 0.0])

    transform = pin.SE3.Identity()
    # transform = pin.SE3(
    #    pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0), np.array([0.3, 0.0, 0.3])
    # )

    pose = pose.act(transform)
    if i % freq == 0:
        print(pose.translation)
    print("translation:", pose.translation)
    print("rotation:", pin.Quaternion(pose.rotation))

    return {"T_goal": pose, "v": np.zeros(6)}


if __name__ == "__main__":
    args = getArgs()

    # command_sender: 7777
    sender = ProcessManager(
        args, server, {"T_goal": pin.SE3.Identity(), "v": np.zeros(6)}, 0
    )
    # wrench_receiver: 6666
    args.port = 6666
    receiver = ProcessManager(
        args, client, {"wrench": np.zeros(6)}, 4, init_value={"wrench": np.zeros(6)}
    )
    freq = 200
    i = 0
    while True:
        i += 1
        sender.sendCommand(sendCircleGoalReceiveWrench(freq, i))
        if i % freq == 0:
            print(receiver.getData())
        time.sleep(1 / 200)

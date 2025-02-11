# PYTHON_ARGCOMPLETE_OK
import numpy as np
import time
import argparse, argcomplete
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager, ProcessManager
from ur_simple_control.optimal_control.get_ocp_args import get_OCP_args
from ur_simple_control.optimal_control.crocoddyl_optimal_control import *
from ur_simple_control.optimal_control.crocoddyl_mpc import *
from ur_simple_control.basics.basics import followKinematicJointTrajP
from ur_simple_control.util.logging_utils import LogManager
from ur_simple_control.visualize.visualize import plotFromDict
from ur_simple_control.clik.clik import getClikArgs, cartesianPathFollowingWithPlanner, controlLoopClik, invKinmQP, dampedPseudoinverse
import pinocchio as pin
import crocoddyl
from functools import partial
import importlib.util
from ur_simple_control.path_generation.planner import starPlanner, getPlanningArgs, createMap
import yaml
import numpy as np
from functools import partial
from ur_simple_control.managers import ProcessManager, getMinimalArgParser
from ur_simple_control.util.get_model import heron_approximation
from ur_simple_control.visualize.visualize import plotFromDict, realTimePlotter, manipulatorVisualizer
from ur_simple_control.path_generation.planner import path2D_timed, pathPointFromPathParam, path2D_to_SE3


def get_args():
    parser = getMinimalArgParser()
    parser = get_OCP_args(parser)
    parser = getClikArgs(parser) # literally just for goal error
    parser = getPlanningArgs(parser)
    parser.add_argument('--handlebar-height', type=float, default=0.5,\
                        help="heigh of handlebar of the cart to be pulled")
    parser.add_argument('--base-to-handlebar-preferred-distance', type=float, default=0.7, \
            help="prefered path arclength from mobile base position to handlebar")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    # TODO TODO TODO: REMOVE PRESET HACKS
    robot_type = "Unicycle"
    with open(args.planning_robot_params_file) as f:
        params = yaml.safe_load(f)
    robot_params = params[robot_type]
    with open(args.tunnel_mpc_params_file) as f:
        params = yaml.safe_load(f)
    mpc_params = params["tunnel_mpc"]
    args.np = mpc_params['np']
    args.n_pol = mpc_params['n_pol']
    return args

args = get_args()
args.past_window_size = 100

model, visual_model, collision_model, data = heron_approximation()
side_function = partial(manipulatorVisualizer, model, collision_model, visual_model)
#side_function = partial(manipulatorVisualizer, None, None, None)
q = np.zeros(model.nq)
q[0] = 10
q[1] = 10
q[2] = 0
q[3] = 1
visualizer_manager = ProcessManager(args, side_function, {"q" : q}, 0)

# s = t * v
# t=2
# v= max_base_v

max_base_v = np.linalg.norm(model.velocityLimit[:2])
dt = 1/ args.ctrl_freq

# path you already traversed
#time_past = np.linspace(0.0, args.past_window_size * dt, args.past_window_size)
x = np.linspace(0.0, args.past_window_size * dt, args.past_window_size)
#x = np.linspace(0.0, 2.0, 200)
x = x.reshape((-1,1))
y = np.sin(x)
past_data  = {}
past_data['path2D_untimed'] = np.hstack((x,y))

# path you get from path planner
x= np.linspace(0.0, args.past_window_size * dt, args.past_window_size)
#x = np.linspace(2.0, 4.0, 200)
x = x.reshape((-1,1))
y = np.sin(x)
path2D_untimed_base = np.hstack((x,y))
p = path2D_untimed_base[-1]
path2D_untimed_base = np.array(path2D_untimed_base).reshape((-1,2))
# ideally should be precomputed somewhere 
# base just needs timing on the path,
# and it's of height 0 (i.e. the height of base's planar joint)
path_base = path2D_timed(args, None, path2D_untimed_base, max_base_v, 0.0)

path_arclength = np.linalg.norm(p - past_data['path2D_untimed'])
handlebar_path_index = -1
for i in range(-2, -1 * len(past_data['path2D_untimed']), -1):
    if path_arclength > args.base_to_handlebar_preferred_distance:
        handlebar_path_index = i
        break
    path_arclength += np.linalg.norm(past_data['path2D_untimed'][i - 1] - past_data['path2D_untimed'][i])
# i shouldn't need to copy-paste everything but what can you do
path2D_handlebar_1_untimed = np.array(past_data['path2D_untimed'])


time_past = np.linspace(0.0, args.past_window_size * dt, args.past_window_size)
s = np.linspace(0.0, args.n_knots * args.ocp_dt, args.n_knots)
path2D_handlebar_1 = np.hstack((
    np.interp(s, time_past, path2D_handlebar_1_untimed[:,0]).reshape((-1,1)), 
    np.interp(s, time_past, path2D_handlebar_1_untimed[:,1]).reshape((-1,1))))

# TODO: combine with base for maximum correctness
pathSE3_handlebar = path2D_to_SE3(path2D_handlebar_1, args.handlebar_height)
for p in pathSE3_handlebar:
    print(p)

#some_path = []
#for i in range(100):
#    translation = np.zeros(3)
#    translation[0] = i / 100
#    rotation = np.eye(3)
#    some_path.append(pin.SE3(rotation, translation))
for i in range(100):
    visualizer_manager.sendCommand({"frame_path": pathSE3_handlebar})
    #visualizer_manager.sendCommand({"frame_path": some_path})
    visualizer_manager.sendCommand({"path": pathSE3_handlebar})
    time.sleep(1)
print("send em")

time.sleep(10)
visualizer_manager.terminateProcess()

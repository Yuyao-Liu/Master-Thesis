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
from ur_simple_control.clik.clik import getClikArgs, cartesianPathFollowingWithPlanner, controlLoopClik, invKinmQP, dampedPseudoinverse, controlLoopClikArmOnly, controlLoopClikDualArmsOnly
import pinocchio as pin
import crocoddyl
from functools import partial
import importlib.util
from ur_simple_control.path_generation.planner import starPlanner, getPlanningArgs, createMap
import yaml

# TODO:
# - make reference step size in path_generator an argument here
#   because we use that for path interpolation later on as well

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

def isGraspOK(args, robot, grasp_pose):
    isOK = False
    SEerror = robot.getT_w_e().actInv(grasp_pose)
    err_vector = pin.log6(SEerror).vector 
    # TODO: figure this out
    # it seems you have to use just the arm to get to finish with this precision 
    #if np.linalg.norm(err_vector) < robot.args.goal_error:
    if np.linalg.norm(err_vector) < 2*1e-1:
        isOK = True
    return not isOK 


def isGripperRelativeToBaseOK(args, robot):
    isOK = False
    # we want to be in the back of the base (x-axis) and on handlebar height
    T_w_base = robot.data.oMi[1]
    # rotation for the gripper is base with z flipped to point into the ground
    rotate = pin.SE3(pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0), np.zeros(3))
    # translation is prefered distance from base
    translate = pin.SE3(np.eye(3), np.array([args.base_to_handlebar_preferred_distance, 0.0, args.handlebar_height]))
    #grasp_pose = T_w_base.act(rotate.act(translate))
    grasp_pose = T_w_base.act(translate.act(rotate))
    SEerror = robot.getT_w_e().actInv(grasp_pose)
    err_vector = pin.log6(SEerror).vector 
    # TODO: figure this out
    # it seems you have to use just the arm to get to finish with this precision 
    #if np.linalg.norm(err_vector) < robot.args.goal_error:
    if np.linalg.norm(err_vector) < 2*1e-1:
        isOK = True
    return isOK, grasp_pose 

def areDualGrippersRelativeToBaseOK(args, goal_transform, robot):
    isOK = False
    # we want to be in the back of the base (x-axis) and on handlebar height
    T_w_base = robot.data.oMi[1]
    # rotation for the gripper is base with z flipped to point into the ground
    rotate = pin.SE3(pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0), np.zeros(3))
    # translation is prefered distance from base
    translate = pin.SE3(np.eye(3), np.array([args.base_to_handlebar_preferred_distance, 0.0, args.handlebar_height]))
    #grasp_pose = T_w_base.act(rotate.act(translate))
    grasp_pose = T_w_base.act(translate.act(rotate))

    grasp_pose_left = goal_transform.act(grasp_pose)
    grasp_pose_right = goal_transform.inverse().act(grasp_pose)

    T_w_e_left, T_w_e_right = robot.getT_w_e()
    SEerror_left = T_w_e_left.actInv(grasp_pose_left)
    SEerror_right = T_w_e_right.actInv(grasp_pose_right)
    err_vector_left = pin.log6(SEerror_left).vector 
    err_vector_right = pin.log6(SEerror_right).vector 
    # TODO: figure this out
    # it seems you have to use just the arm to get to finish with this precision 
    #if np.linalg.norm(err_vector) < robot.args.goal_error:
    if (np.linalg.norm(err_vector_left) < 2*1e-1) and (np.linalg.norm(err_vector_right) < 2*1e-1):
        isOK = True
    return isOK, grasp_pose, grasp_pose_left, grasp_pose_right

def cartPullingControlLoop(args, robot : RobotManager, goal, goal_transform, solver_grasp, solver_pulling,
                           path_planner : ProcessManager, i : int, past_data):
    """
    cartPulling
    0) if obstacles present, don't move
    1) if no obstacles, but not grasped/grasp is off-target, grasp the handlebar with a p2p strategy.
    2) if no obstacles, and grasp ok, then pull toward goal with cartPulling mpc
    3) parking?
    4) ? always true (or do nothing is always true, whatever)
    """

    q = robot.getQ()
    if robot.robot_name != "yumi":
        T_w_e = robot.getT_w_e()
    else:
        T_w_e_left, T_w_e_right = robot.getT_w_e()

    # we use binary as string representation (i don't want to deal with python's binary representation).
    # the reason for this is that then we don't have disgusting nested ifs
    # TODO: this has to have named entries for code to be readable
    priority_register = ['0','1','1']
    # TODO implement this based on laser scanning or whatever
    #priority_register[0] = str(int(areObstaclesTooClose()))
    if robot.robot_name != "yumi":
        graspOK, grasp_pose = isGripperRelativeToBaseOK(args, robot)
    else:
        graspOK, grasp_pose, grasp_pose_left, grasp_pose_right = areDualGrippersRelativeToBaseOK(args, goal_transform, robot)
    # NOTE: this keeps getting reset after initial grasp has been completed.
    # and we want to let mpc cook
    priority_register[1] = str(int(not graspOK)) # set if not ok
    # TODO: get grasp pose from vision, this is initial hack to see movement
#    if i > 1000:
#        priority_register[1] = '0'
    
    # interpret string as base 2 number, return int in base 10
    priority_int = ""
    for prio in priority_register:
        priority_int += prio
    priority_int = int(priority_int, 2)
    breakFlag = False
    save_past_item = {}
    log_item = {}

    # case 0)
    if priority_int >= 4:
        robot.sendQd(np.zeros(robot.model.nv))

    # case 1)
    # TODO: make it an argument obviously
    usempc = False
    if (priority_int < 4) and (priority_int >= 2):
        # TODO: make goal an argument, remove Mgoal from robot
        robot.Mgoal = grasp_pose
        if usempc:
            for i, runningModel in enumerate(solver_grasp.problem.runningModels):
                if robot.robot_name != "yumi":
                    runningModel.differential.costs.costs['gripperPose'].cost.residual.reference = grasp_pose
                else:
                    runningModel.differential.costs.costs['gripperPose_l'].cost.residual.reference = grasp_pose_left
                    runningModel.differential.costs.costs['gripperPose_r'].cost.residual.reference = grasp_pose_right
            if robot.robot_name != "yumi":
                solver_grasp.problem.terminalModel.differential.costs.costs['gripperPose'].cost.residual.reference = grasp_pose
            else:
                solver_grasp.problem.terminalModel.differential.costs.costs['gripperPose_l'].cost.residual.reference = grasp_pose_left
                solver_grasp.problem.terminalModel.differential.costs.costs['gripperPose_r'].cost.residual.reference = grasp_pose_right

            robot.Mgoal = grasp_pose
            #breakFlag, save_past_item, log_item = CrocoIKMPCControlLoop(args, robot, solver_grasp, i, past_data)
            CrocoIKMPCControlLoop(args, robot, solver_grasp, i, past_data)
        else:
            #controlLoopClik(robot, invKinmQP, i, past_data)
            clikController = partial(dampedPseudoinverse, 1e-3)
            #controlLoopClik(robot, clikController, i, past_data)
            if robot.robot_name != "yumi":
                controlLoopClikArmOnly(robot, clikController, i, past_data)
            else:
                # TODO: DEFINE SENSIBLE TRANSFOR
                controlLoopClikDualArmsOnly(robot, clikController, goal_transform, i, past_data)

    # case 2)
    # MASSIVE TODO: 
    # WHEN STARTING, TO INITIALIZE PREPOPULATE THE PATH WITH AN INTERPOLATION OF 
    # A LINE FROM WHERE THE GRIPPER IS NOW TO THE BASE

    # whether we're transitioning from cliking to pulling
    # this is the time and place to populate the past path from the pulling to make sense
    if past_data['priority_register'][-1][1] == '1' and priority_register[1] == '0': 
        # create straight line path from ee to base
        p_cart = q[:2]
        if robot.robot_name != "yumi":
            p_ee = T_w_e.translation[:2]
            straigh_line_path = np.linspace(p_ee, p_cart, args.past_window_size)
        else:
            p_ee_l = T_w_e_left.translation[:2]
            p_ee_r = T_w_e_right.translation[:2]
        # MASSIVE TODO: have two different REFERENCES FOR TWO ARMS
        # EVILLLLLLLLLLLLLLLLLLLLLLLLLLll
        # MASSIVE TODO: have two different REFERENCES FOR TWO ARMS
        # MASSIVE TODO: have two different REFERENCES FOR TWO ARMS
        # MASSIVE TODO: have two different REFERENCES FOR TWO ARMS
# ----------------> should be doable by just hitting the path with the goal_transform for dual.
#                   in the whole task there are no differences in left and right arm
            straigh_line_path = np.linspace(p_ee_l, p_cart, args.past_window_size)
        # time it the same way the base path is timed 
        time_past = np.linspace(0.0, args.past_window_size * robot.dt, args.past_window_size)
        s = np.linspace(0.0, args.n_knots * args.ocp_dt, args.past_window_size)
        path2D_handlebar = np.hstack((
            np.interp(s, time_past, straigh_line_path[:,0]).reshape((-1,1)), 
            np.interp(s, time_past, straigh_line_path[:,1]).reshape((-1,1))))

        past_data['path2D_untimed'].clear()
        past_data['path2D_untimed'].extend(path2D_handlebar[i] for i in range(args.past_window_size))

    if priority_int < 2:
        # TODO make this one work
        breakFlag, save_past_item, log_item = BaseAndEEPathFollowingMPCControlLoop(args, robot, solver_pulling, path_planner, grasp_pose, goal_transform, i, past_data)
        #BaseAndEEPathFollowingMPCControlLoop(args, robot, solver_pulling, path_planner, i, past_data)

    p = q[:2]
    # needed for cart pulling
    save_past_item['path2D_untimed'] = p
    save_past_item['priority_register'] = priority_register.copy()
    # TODO plot priority register out
    #log_item['prio_reg'] = ...
    log_item['qs'] = q.reshape((robot.model.nq,))
    log_item['dqs'] = robot.getQd().reshape((robot.model.nv,))
    print(priority_register)
    return breakFlag, save_past_item, log_item


def cartPulling(args, robot : RobotManager, goal, path_planner):
    # transfer single arm to dual arm reference
    # (instead of gripping the middle grip slightly away from middle)
    goal_transform = pin.SE3.Identity()
# TODO: it's cursed and it shouldn't be
#    goal_transform.rotation = pin.rpy.rpyToMatrix(0.0, np.pi/2, 0.0)
# TODO: it's cursed and it shouldn't be
#    goal_transform.translation[1] = -0.1
    ############################
    #  setup cart-pulling mpc  #
    ############################
    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    problem_pulling = createBaseAndEEPathFollowingOCP(args, robot, x0)
    if args.solver == "boxfddp":
        solver_pulling = crocoddyl.SolverBoxFDDP(problem_pulling)
    if args.solver == "csqp":
        solver_pulling = mim_solvers.SolverCSQP(problem_pulling)
    xs_init = [x0] * (solver_pulling.problem.T + 1)
    us_init = solver_pulling.problem.quasiStatic([x0] * solver_pulling.problem.T)
    solver_pulling.solve(xs_init, us_init, args.max_solver_iter)

    #############################################
    #  setup point-to-point handlebar grasping  #
    # TODO: have option to swith this for clik  #
    #############################################
    grasp_pose = robot.getT_w_e()
    problem_grasp = createCrocoIKOCP(args, robot, x0, grasp_pose)
    if args.solver == "boxfddp":
        solver_grasp = crocoddyl.SolverBoxFDDP(problem_grasp)
    if args.solver == "csqp":
        solver_grasp = mim_solvers.SolverCSQP(problem_grasp)
    xs_init = [x0] * (solver_grasp.problem.T + 1)
    us_init = solver_grasp.problem.quasiStatic([x0] * solver_grasp.problem.T)
    solver_grasp.solve(xs_init, us_init, args.max_solver_iter)
    
    controlLoop = partial(cartPullingControlLoop, args, robot, goal, goal_transform, solver_grasp, solver_pulling, path_planner)

    log_item = {}
    q = robot.getQ()
    if robot.robot_name != "yumi":
        T_w_e = robot.getT_w_e()
    else:
        T_w_e_l, T_w_e_right = robot.getT_w_e()
    log_item['qs'] = q.reshape((robot.model.nq,))
    log_item['dqs'] = robot.getQd().reshape((robot.model.nv,))
    #T_base = self.robot_manager.data.oMi[1]
    # NOTE: why the fuck was the past path defined from the end-effector?????
    #save_past_item = {'path2D_untimed' : T_w_e.translation[:2],
    save_past_item = {'path2D_untimed' : q[:2],
                      'priority_register' : ['0','1','1']}
    loop_manager = ControlLoopManager(robot, controlLoop, args, save_past_item, log_item)
    loop_manager.run()

    

if __name__ == "__main__": 
    args = get_args()
    if importlib.util.find_spec('mim_solvers'):
        import mim_solvers
    robot = RobotManager(args)
    robot.q[0] = 9.0
    robot.q[1] = 4.0

    x0 = np.concatenate([robot.getQ(), robot.getQd()])
    robot._step()
    goal = np.array([0.5, 5.5])

    ###########################
    #  visualizing obstacles  #
    ###########################
    _, map_as_list = createMap()
    # we're assuming rectangles here
    # be my guest and implement other options
    if args.visualize_manipulator:
        for obstacle in map_as_list:
            length = obstacle[1][0] - obstacle[0][0]
            width =  obstacle[3][1] - obstacle[0][1]
            height = 0.4 # doesn't matter because plan because planning is 2D
            pose = pin.SE3(np.eye(3), np.array([
               obstacle[0][0] + (obstacle[1][0] - obstacle[0][0]) / 2,
               obstacle[0][1] + (obstacle[3][1] - obstacle[0][1]) / 2, 
                0.0]))
            dims = [length, width, height]
            command = {"obstacle" : [pose, dims]}
            robot.visualizer_manager.sendCommand(command)

    planning_function = partial(starPlanner, goal)
    # TODO: ensure alignment in orientation between planner and actual robot
    path_planner = ProcessManager(args, planning_function, robot.q[:2], 3, None)
    # wait for meshcat to initialize
    if args.visualize_manipulator:
        time.sleep(5)
    # clik version
    #cartesianPathFollowingWithPlanner(args, robot, path_planner)
    # end-effector tracking version
    #CrocoEndEffectorPathFollowingMPC(args, robot, x0, path_planner)
    # base tracking version (TODO: implement a reference for ee too)
    # and also make the actual path for the cart and then construct the reference
    # for the mobile base out of a later part of the path)
    # atm this is just mobile base tracking 
    cartPulling(args, robot, goal, path_planner)
    print("final position:")
    print(robot.getT_w_e())
    path_planner.terminateProcess()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if not args.pinocchio_only:
        robot.stopRobot()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()
    #loop_manager.stopHandler(None, None)


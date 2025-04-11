from smc.control.control_loop_manager import ControlLoopManager
from smc.control.controller_templates.point_to_point import DualEEP2PCtrlLoopTemplate
from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.robots.interfaces.dual_arm_interface import DualArmInterface

import pink
from pink.barriers import BodySphericalBarrier
from pink.tasks import FrameTask, PostureTask

import numpy as np
import qpsolvers
import argparse
from functools import partial
from collections import deque
import pinocchio as pin
from smc.robots.interfaces.mobile_base_interface import MobileBaseInterface


# TODO: butcher pink to avoid redundancies with smc like configuration.
# right now there is no time and we're just shoving it in.
def DualArmIKSelfAvoidanceViaEndEffectorSpheresCtrlLoop(
    tasks: list[pink.FrameTask],
    cbf_list: list[pink.barriers.Barrier],
    solver: str,
    T_w_absgoal: pin.SE3,
    T_absgoal_l: pin.SE3,
    T_absgoal_r: pin.SE3,
    args: argparse.Namespace,
    robot: DualArmInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:

    # NOTE: for control modes to work, passing a reduced model is required.
    # this model takes in full q, not the reduced one
    # TODO: re-write stuff to accomodate new model
    configuration = pink.Configuration(robot.model, robot.data, robot._q)
    # NOTE: there are limits in pink, but they are a class where G matrix is constucted
    # combining lb and ub and forming Gx \leq h
    # this makes it possible to combine other stuff into this inequality constraint
    configuration.model.velocityLimit = robot._max_v

    T_w_lgoal = T_absgoal_l.act(T_w_absgoal)
    T_w_rgoal = T_absgoal_r.act(T_w_absgoal)
    # next_goal_l = pin.SE3.Interpolate(robot.T_w_l, T_w_lgoal, 0.001)
    # next_goal_r = pin.SE3.Interpolate(robot.T_w_r, T_w_rgoal, 0.001)
    # tasks[0].set_target(next_goal_l)
    # tasks[1].set_target(next_goal_r)
    tasks[0].set_target(T_w_lgoal)
    tasks[1].set_target(T_w_rgoal)

    v_cmd = pink.solve_ik(
        configuration,
        tasks,
        robot.dt,
        solver=solver,
        barriers=cbf_list,
        # safety_break=True,
        safety_break=False,
    )
    # NOTE: this is a temporary solution to deal with the fact that model isn't a property depending on control mode yet
    # TODO: make model a property depending on control mode to avoid this shitty issue
    if robot.mode == AbstractRobotManager.control_mode.upper_body:
        # v_cmd[:3] = 0.0
        v_cmd = v_cmd[3:]
    dist_ee = np.linalg.norm(robot.T_w_l.translation - robot.T_w_r.translation)
    log_item = {"dist_ees": dist_ee.reshape((1,))}
    return v_cmd, {}, log_item


def DualArmIKSelfAvoidanceViaEndEffectorSpheres(
    T_w_absgoal: pin.SE3,
    T_absgoal_l: pin.SE3,
    T_absgoal_r: pin.SE3,
    args: argparse.Namespace,
    robot: DualArmInterface,
    run: bool = True,
) -> ControlLoopManager | None:

    # Pink tasks
    left_end_effector_task = FrameTask(
        "robl_tool0",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )
    right_end_effector_task = FrameTask(
        "robr_tool0",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )

    # Pink barriers
    ee_barrier = BodySphericalBarrier(
        ("robl_tool0", "robr_tool0"),
        d_min=0.15,
        gain=100.0,
        safe_displacement_gain=1.0,
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    cbf_list = [ee_barrier]
    tasks = [left_end_effector_task, right_end_effector_task, posture_task]

    # NOTE: model and data are shared pointers between configuration and robot.
    # this is redundant as hell, but I don't have the time butcher pink right now
    # NOTE: for control modes to work, passing a reduced model is required.
    # this model takes in full q, not the reduced one
    # TODO: re-write stuff to accomodate new model
    configuration = pink.Configuration(robot.model, robot.data, robot._q)
    posture_task.set_target_from_configuration(configuration)
    left_end_effector_task.set_target(T_absgoal_l.act(T_w_absgoal))
    right_end_effector_task.set_target(T_absgoal_r.act(T_w_absgoal))

    #    meshcat_shapes.sphere(
    #        viewer["left_ee_barrier"],
    #        opacity=0.4,
    #        color=0xFF0000,
    #        radius=0.125,
    #    )
    #    meshcat_shapes.sphere(
    #        viewer["right_ee_barrier"],
    #        opacity=0.4,
    #        color=0x00FF00,
    #        radius=0.125,
    #    )
    #    meshcat_shapes.frame(viewer["left_end_effector_target"], opacity=1.0)
    #    meshcat_shapes.frame(viewer["right_end_effector_target"], opacity=1.0)

    # TODO: allow proxsuite solvers also (primarily because they can be saved and reused with warmstarting)
    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    ctrl_loop = partial(
        DualArmIKSelfAvoidanceViaEndEffectorSpheresCtrlLoop,
        tasks,
        cbf_list,
    )
    control_loop = partial(
        DualEEP2PCtrlLoopTemplate,
        solver,
        T_w_absgoal,
        T_absgoal_l,
        T_absgoal_r,
        ctrl_loop,
        args,
        robot,
    )
    log_item = {
        "qs": robot.q,
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros((robot.model.nv,)),
        "l_err_norm": np.zeros((1,)),
        "r_err_norm": np.zeros((1,)),
        "dist_ees": np.zeros((1,)),
    }
    loop_manager = ControlLoopManager(robot, control_loop, args, {}, log_item)
    if not run:
        return loop_manager
    else:
        loop_manager.run()

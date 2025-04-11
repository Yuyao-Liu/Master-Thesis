from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.control.control_loop_manager import ControlLoopManager

import pinocchio as pin
import numpy as np
from functools import partial
from collections import deque


# NOTE: it's probably a good idea to generalize this for different references:
# - only qs
# - qs and vs
# - whatever special case
# it could be better to have separate functions, whatever makes the code as readable
# as possible.
# also NOTE: there is an argument to be made for pre-interpolating the reference.
# this is because joint positions will be more accurate.
# if we integrate them with interpolated velocities.
def followKinematicJointTrajPControlLoop(
    stop_at_final: bool,
    robot: AbstractRobotManager,
    reference,
    i: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]:
    breakFlag = False
    save_past_dict = {}
    log_item = {}
    q = robot.q
    v = robot.v
    # NOTE: assuming we haven't missed a timestep,
    # which is pretty much true
    t = i * robot.dt
    # take the future (next) one
    t_index_lower = int(np.floor(t / reference["dt"]))
    t_index_upper = int(np.ceil(t / reference["dt"]))

    # TODO: set q_refs and v_refs once (merge with interpolation if)
    if t_index_upper >= len(reference["qs"]) - 1:
        t_index_upper = len(reference["qs"]) - 1
    q_ref = reference["qs"][t_index_upper]
    if (t_index_upper == len(reference["qs"]) - 1) and stop_at_final:
        v_ref = np.zeros(robot.model.nv)
    else:
        v_ref = reference["vs"][t_index_upper]

    # TODO NOTE TODO TODO: move under stop/don't stop at final argument
    if (
        (t_index_upper == len(reference["qs"]) - 1)
        and (np.linalg.norm(q - reference["qs"][-1]) < 1e-2)
        and (np.linalg.norm(v) < 1e-2)
    ):
        breakFlag = True

    # TODO: move interpolation to a different function later
    # --> move it to a module called math
    if (t_index_upper < len(reference["qs"]) - 1) and (not breakFlag):
        # angle = (reference['qs'][t_index_upper] - reference['qs'][t_index_lower]) / reference['dt']
        angle_v = (
            reference["vs"][t_index_upper] - reference["vs"][t_index_lower]
        ) / reference["dt"]
        time_difference = t - t_index_lower * reference["dt"]
        v_ref = reference["vs"][t_index_lower] + angle_v * time_difference
        # using pin.integrate to make this work for all joint types
        # NOTE: not fully accurate as it could have been integrated with previous interpolated velocities,
        # but let's go for it as-is for now
        # TODO: make that work via past_data IF this is still too saw-looking
        q_ref = pin.integrate(
            robot.model,
            reference["qs"][t_index_lower],
            reference["vs"][t_index_lower] * time_difference,
        )

    error_q = pin.difference(robot.model, q, q_ref)
    error_v = v_ref - v
    Kp = 1.0
    Kd = 0.5

    #          feedforward                      feedback
    v_cmd = v_ref + Kp * error_q  # + Kd * error_v
    # qd_cmd = v_cmd[:6]
    robot.sendVelocityCommand(v_cmd)

    log_item["error_qs"] = error_q
    log_item["error_vs"] = error_v
    log_item["qs"] = q
    log_item["vs"] = v
    log_item["vs_cmd"] = v_cmd
    log_item["reference_qs"] = q_ref
    log_item["reference_vs"] = v_ref

    return breakFlag, {}, log_item


def followKinematicJointTrajP(args, robot, reference):
    # we're not using any past data or logging, hence the empty arguments
    controlLoop = partial(
        followKinematicJointTrajPControlLoop, args.stop_at_final, robot, reference
    )
    log_item = {
        "error_qs": np.zeros(robot.model.nv),  # differences live in the tangent space
        "error_vs": np.zeros(robot.model.nv),
        "qs": np.zeros(robot.model.nq),
        "vs": np.zeros(robot.model.nv),
        "vs_cmd": np.zeros(robot.model.nv),
        "reference_qs": np.zeros(robot.model.nq),
        "reference_vs": np.zeros(robot.model.nv),
    }
    loop_manager = ControlLoopManager(robot, controlLoop, args, {}, log_item)
    loop_manager.run()
    if args.debug_prints:
        print("followKinematicJointTrajP done: reached path destionation!")

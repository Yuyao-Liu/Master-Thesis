from smc.path_generation.path_math.path_to_trajectory import path2D_to_trajectory2D
from smc.path_generation.path_math.path2d_to_6d import (
    path2D_to_SE3,
)

import numpy as np
import pinocchio as pin
from argparse import Namespace
from collections import deque


# NOTE: extremely inefficient.
# there should be no copy-pasting of the whole path at every single point in time,
# instead all the computations should be cached.
# ideally you don't even cast the whole past_path into an array, but work on the queue,
# and just do the smallest update.
# this shouldn't be too difficult, it's just annoying. first we fix the logic,
# then we improve efficiency.
def getCurrentPositionHandlebarInPastBasePath(
    args: Namespace, p_base_current: np.ndarray, past_path2D_untimed: np.ndarray
) -> int:
    """
    getCurrentPositionHandlebarInPastBasePath
    -------------------------------------------
    0. paths points are 2D [x,y] coordinates in the world frame
    1. past path goes from oldest entry to newest in indeces
    2. we want a point on the path that's args.base_to_handlebar_preferred_distance in arclength
    3. by minusing the whole path from the current point, we get relative distances of the path
       points from the current point
    4. to make life easier, we linearly interpolate between the path point.
    5. we add the lengths of lines between path points until we reach args.base_to_handlebar_preferred_distance
    6. we return the index of the path point to make further math easier

    NOTE: this can be O(1) instead of O(n) but i can't be bothered
          (i could save arclengths instead of re-calculating them all the time, but the cost
          of the algo as it is now is not high enough to justify this optimization)
    """
    arclength = 0.0
    path_relative_distances = np.linalg.norm(
        p_base_current - past_path2D_untimed, axis=1
    )
    handlebar_path_index = (
        0  # index of sought-after point, 0 is the furthest in the past
    )
    # go through whole path backwards
    for i in range(-2, -1 * len(past_path2D_untimed), -1):
        if arclength > args.base_to_handlebar_preferred_distance:
            handlebar_path_index = i
            return handlebar_path_index
        arclength += np.linalg.norm(
            path_relative_distances[i - 1] - path_relative_distances[i]
        )
    print(
        "the size of the past path is too short to find the point of prefered distance on it!"
    )
    print("i'll just give you the furtherst point, which is of distance", arclength)
    return handlebar_path_index


# NOTE: extremely inefficient.
# there should be no copy-pasting of the whole path at every single point in time,
# instead all the computations should be cached.
# ideally you don't even cast the whole past_path into an array, but work on the queue,
# and just do the smallest update.
# this shouldn't be too difficult, it's just annoying. first we fix the logic,
# then we improve efficiency.
def construct_EE_path(
    args: Namespace,
    p_base_current: np.ndarray,
    past_path2D_from_window: deque[np.ndarray],
) -> list[pin.SE3]:
    """
    construct_EE_path
    -----------------
    - path_past2D - should be this entry in past rolling window, turned into a numpy array and it's length should be args.past_window_size.
                    just in case, and for potential transferability, we will use len(path_past2D) here

    ###################################################
    #  construct timed SE3 path for the end-effector  #
    ###################################################
    this works as follow:
        1) find the previous path point of arclength base_to_handlebar_preferred_distance.
        first part of the path is from there to current base position,
        second is just the current base's plan.
        2) construct timing on the first part. (robot.dt =/= ocp_dt)
        3) (optional) join that with the already timed second part.
            --> not doing this because paths are super short and this won't ever happen with the current setup
        4) turn the timed 2D path into an SE3 trajectory

    (1) NOTE: BIG ASSUMPTION
        let's say we're computing on time, and that's the time spacing
        of previous path points.
        this means you need to lower the control frequency argument
        if you're not meeting deadlines.
        if you really need timing information, you should actually
        get it from ControlLoopManager instead of i,
        but let's say this is better because you're forced to know
        how fast you are instead of ducktaping around delays.
    TODO: actually save timing, pass t instead of i to controlLoops
        from controlLoopManager
    NOTE: this might not working when rosified
    """
    # i shouldn't need to copy-paste everything but what can you do
    past_path2D = np.array(past_path2D_from_window).reshape((-1, 2))
    # print("-" * 5, "past_window", "-" * 5)
    # print(past_path2D)
    # step (1)
    handlebar_path_index = getCurrentPositionHandlebarInPastBasePath(
        args, p_base_current, past_path2D
    )
    # cut of the past that's irrelevant
    ee_path = past_path2D[handlebar_path_index:]
    path_len = len(ee_path)
    # step (2)
    # look at note (1)
    time_past = np.linspace(0.0, path_len * (1 / args.ctrl_freq), path_len)
    s = np.linspace(0.0, args.n_knots * args.ocp_dt, args.n_knots)
    ee_2Dtrajectory = np.hstack(
        (
            np.interp(s, time_past, ee_path[:, 0]).reshape((-1, 1)),
            np.interp(s, time_past, ee_path[:, 1]).reshape((-1, 1)),
        )
    )

    # step (4)
    ee_SE3trajectory = path2D_to_SE3(ee_2Dtrajectory, args.handlebar_height)
    return ee_SE3trajectory

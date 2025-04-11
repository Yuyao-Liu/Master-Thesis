import numpy as np
from argparse import Namespace
from pinocchio import SE3


def computePath2DLength(path2D):
    x_i = path2D[:, 0][:-1]  # no last element
    y_i = path2D[:, 1][:-1]  # no last element
    x_i_plus_1 = path2D[:, 0][1:]  # no first element
    y_i_plus_1 = path2D[:, 1][1:]  # no first element
    x_diff = x_i_plus_1 - x_i
    x_diff = x_diff.reshape((-1, 1))
    y_diff = y_i_plus_1 - y_i
    y_diff = y_diff.reshape((-1, 1))
    path_length = np.sum(np.linalg.norm(np.hstack((x_diff, y_diff)), axis=1))
    return path_length


def path2D_to_trajectory2D(
    args: Namespace, path2D: np.ndarray, velocity: float
) -> np.ndarray:
    """
    path2D_to_trajectory2D
    ---------------------
    read the technical report for details
    """
    path_length = computePath2DLength(path2D)
    # NOTE: sometimes the path planner gives me the same god damn points
    # and that's it. can't do much about, expect set those point then.
    # and i need to return now so that the math doesn't break down the line
    if path_length < 1e-3:
        return np.ones((args.n_knots + 1, 2)) * path2D[0]
    total_time = path_length / velocity
    # NOTE: assuming the path is uniformly sampled
    t_path = np.linspace(0.0, total_time, len(path2D))
    t_ocp = np.linspace(0.0, args.ocp_dt * (args.n_knots + 1), args.n_knots + 1)

    trajectory2D = np.array(
        [
            np.interp(t_ocp, t_path, path2D[:, 0]),
            np.interp(t_ocp, t_path, path2D[:, 1]),
        ]
    ).T
    return trajectory2D


def pathSE3_to_trajectorySE3(
    args: Namespace, pathSE3: list[SE3], velocity: float
) -> list[SE3]:
    path2D = np.zeros((len(pathSE3), 2))
    for i, pose in enumerate(pathSE3):
        path2D[i] = pose.translation[:2]
    path_length = computePath2DLength(path2D)
    # NOTE: sometimes the path planner gives me the same god damn points
    # and that's it. can't do much about, expect set those point then.
    # and i need to return now so that the math doesn't break down the line
    if path_length < 1e-3:
        return [pathSE3[0]] * (args.n_knots + 1)
    total_time = path_length / velocity
    # NOTE: assuming the path is uniformly sampled
    t_path = np.linspace(0.0, total_time, len(path2D))
    t_ocp = np.linspace(0.0, args.ocp_dt * (args.n_knots + 1), args.n_knots + 1)

    trajectorySE3 = []
    path_index = 0
    for t_traj in t_ocp:
        if t_traj > t_path[path_index + 1]:
            if path_index < len(pathSE3) - 2:
                path_index += 1

        if path_index < len(pathSE3) - 2:
            pose_traj = SE3.Interpolate(
                pathSE3[path_index],
                pathSE3[path_index + 1],
                t_traj - t_path[path_index],
            )
            trajectorySE3.append(pose_traj)
        else:
            trajectorySE3.append(pathSE3[-1])
    return trajectorySE3

import pinocchio as pin
import numpy as np


def path2D_to_SE3(path2D: np.ndarray, path_height: float) -> list[pin.SE3]:
    """
    path2D_SE3
    ----------
    we have a 2D path of (N,2) shape as reference
    the OCP accepts SE3 (it could just rotations or translation too),
    so this function constructs it from the 2D path.
    """
    #########################
    #  path2D into pathSE2  #
    #########################
    # construct theta
    # since it's a pairwise operation it can't be done on the last point
    x_i = path2D[:, 0][:-1]  # no last element
    y_i = path2D[:, 1][:-1]  # no last element
    x_i_plus_1 = path2D[:, 0][1:]  # no first element
    y_i_plus_1 = path2D[:, 1][1:]  # no first element
    x_diff = x_i_plus_1 - x_i
    y_diff = y_i_plus_1 - y_i
    # elementwise arctan2
    # should be y first, then x
    # thetas = np.arctan2(y_diff, x_diff)
    thetas = np.arctan2(x_diff, y_diff)

    ######################################
    #  construct SE3 from SE2 reference  #
    ######################################
    # the plan is parallel to the ground because it's a mobile
    # manipulation task
    pathSE3 = []
    for i in range(len(path2D) - 1):
        # first set the x axis to be in the theta direction
        # TODO: make sure this one makes sense
        rotation = np.array(
            [
                [np.cos(thetas[i]), np.sin(thetas[i]), 0.0],
                [np.sin(thetas[i]), -np.cos(thetas[i]), 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        # rotation = pin.rpy.rpyToMatrix(0.0, 0.0, thetas[i])
        # rotation = pin.rpy.rpyToMatrix(np.pi / 2, np.pi / 2, 0.0) @ rotation
        translation = np.array([path2D[i][0], path2D[i][1], path_height])
        pathSE3.append(pin.SE3(rotation, translation))
    pathSE3.append(pin.SE3(rotation, translation))
    return pathSE3


# stupid function for stupid data re-assembly
# def path2D_to_SE2(path2D : list):
#    path2D = np.array(path2D)
#    # create (N,2) path for list [x0,y0,x1,y1,...]
#    # of course it shouldn't be like that to begin with but
#    # i have no time for that
#    path2D = path2D.reshape((-1, 2))
#    # since it's a pairwise operation it can't be done on the last point
#    x_i = path2D[:,0][:-1] # no last element
#    y_i = path2D[:,1][:-1] # no last element
#    x_i_plus_1 = path2D[:,0][1:] # no first element
#    y_i_plus_1 = path2D[:,1][1:] # no first element
#    # elementwise arctan2
#    thetas = np.arctan2(x_i_plus_1 - x_i, y_i_plus_1 - y_i)
#    thetas = thetas.reshape((-1, 1))
#    path_SE2 = np.hstack((path2D, thetas))
#    return path_SE2

import numpy as np

#######################################################################
#                    map the pixels to a 3D plane                     #
#######################################################################
def map2DPathTo3DPlane(path_points_2D, width, height):
    """
    map2DPathTo3DPlane
    --------------------
    TODO: THINK AND FINALIZE THE FRAME
    TODO: WRITE PRINT ABOUT THE FRAME TO THE USER
    assumptions:
    - origin in top-left corner (natual for western latin script writing)
    - x goes right (from TCP)
    - z will go away from the board
    - y just completes the right-hand frame
    TODO: RIGHT NOW we don't have a right-handed frame lmao, change that where it should be
    NOTE: this function as well be in the util or drawing file, but whatever for now, it will be done
          once it will actually be needed elsewhere
    Returns a 3D path appropriately scaled, and placed into the first quadrant
    of the x-y plane of the body-frame (TODO: what is the body frame if we're general?)
    """
    z = np.zeros((len(path_points_2D),1))
    path_points_3D = np.hstack((path_points_2D,z))
    # scale the path to m
    path_points_3D[:,0] = path_points_3D[:,0] * width
    path_points_3D[:,1] = path_points_3D[:,1] * height
    # in the new coordinate system we're going in the -y direction
    # TODO this is a demo specific hack, 
    # make it general for a future release
    path_points_3D[:,1] = -1 * path_points_3D[:,1] + height
    return path_points_3D



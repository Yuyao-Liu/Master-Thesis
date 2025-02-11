"""control_test controller."""

from robot_stuff.forw_kinm import *
from robot_stuff.inv_kinm import *
#from anim_func import *
import numpy as np
#import matplotlib.pyplot as plt 
#import mpl_toolkits.mplot3d.axes3d as p3
#from matplotlib.animation import FuncAnimation
#import matplotlib.colors as colr
import sys
import subprocess
import scipy.optimize
import random



# angle in radians ofc
def turnAngleToPlusMinusPiRange(angle):
    return np.arcsin(np.sin(angle))
# let's trace out a circle
# th circle formula is:
# x = radius * cos(t)
# y = radius * sin(t)
# z = height
# the direction you are to move in in order to trace a circle,
# or any parametric curve for that matter is
# in the direction of the derivative at the given point
# the derivative w.r.t. t is, by coordinate:
# x = radius * -1 * sin(t)
# y = radius * cos(t)
# z = 0
def goInACirleViaDerivative(radius, height, current_parameter):
    return np.array([radius * -1 * np.sin(curve_parameter), radius * np.cos(curve_parameter), 0], dtype=float32)


# or just pass the next point to reach and use the vector to get there lel
def goInACirleViaPositionAroundZ(radius, height, current_parameter):
    return np.array([radius * np.cos(curve_parameter), radius * np.sin(curve_parameter), height], dtype=np.float32)

# here height is distance from yz plane
# but also from xy plane (we want it to be floating around axis parallel to x axis actually)
def goInACirleViaPositionAroundLiftedX(radius, height, x_0, current_parameter):
    return np.array([x_0, radius * np.sin(curve_parameter), height
                + radius * np.cos(curve_parameter)], dtype=np.float32)


def error_test(robot, t):
    e = abs(t - r.p_e)
    if e[0] < 0.001 and e[1] < 0.001 and e[2] < 0.001:
        return True
    else:
        return False

# let radius be 0.4
def goInACircleViaJacobian(circle_param):
    """ doing a fixed circle with axis around x axis because
    i only need one circle and this is easy
    arg: s - the current parameter in the parametric eq."""
    # R0c is the rotational matrix for the circle
    # p is the vector to the circle center

    # need to have a hom. transf. mat. to circle center
    # first 3 entries are rot. mat. and last one is translation vec.
    # with added hom. mat. padding
    circ_hom_mat = np.array([[0.0, 0.0, -1.0, 0.0], \
                            [0.0, -1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [-0.59, 0.0, 0.59, 1.0]], dtype=np.float32)
    center = circ_hom_mat[0:3, 3]
    R0c = circ_hom_mat[0:3, 0:3]

    # shouldn't jacobian be a 3x3 mat? 
    circle_jacobian_circle = np.array([-1 * np.sin(circle_param), np.cos(circle_param), 0], dtype=np.float32)
    circle_jacobian_base = R0c @ circle_jacobian_circle

    # now we need a radij-vector
    # you need circle center coordinates and some point on the circle
    # i can calculate p from current parameter value 
    # i do that in circle's frame
    # and then i can transport it to base frame with my hom. transform

    point_on_c_circle = np.array([np.cos(circle_param), np.sin(circle_param), 0.0], dtype=np.float32)
    point_on_c_circle = np.hstack((point_on_c_circle, 1))
    point_on_c_base = circ_hom_mat @ point_on_c_circle

    # need to make center hom.
    center = np.hstack((center, 0))
    radij_vec = point_on_c_base - center

    radius = 0.4

    s =  radius * np.arctan2(radij_vec[1], radij_vec[0])

    s = turnAngleToPlusMinusPiRange(s) + np.pi - s 
    es = s * 2 * np.pi * radius - s

    e = es * circle_jacobian_base
    return e
    



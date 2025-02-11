"""
CREDIT: jnrh2023 pinocchio tutorial

Optimization of the shape of an ellipse so that it contains a set of 3d points.
Used to construct ellipses around manipulator links to get convex self-collision checking.
NOTE: probably cylinders are better but let's run with this.
decide:
 - w in so3: ellipse orientation
 - r in R^3: ellipse main dimensions
minimizing:
  r1*r2*r3 the volum of the ellipse -> possibly not really (can calculate but eh) but it leads to the same result
so that:
  r>=0
  for all points x in a list, x in ellipse: (x - c) @ A @ (x - c) <= 1

with A,c the matrix representation of the ellipsoid A=exp(w)@diag(1/r**2)@exp(w).T

Once solved, this is stored so it doesn't need to be recomputed every the time.
"""

# BIG TODO: make gripper one big ellipse
# going through every link is both wrong and pointlessly complex
# also TODO: try fitting cyllinders instead
import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from ur_simple_control.visualize.meshcat_viewer_wrapper.visualizer import MeshcatVisualizer
from ur_simple_control.managers import RobotManager
from ur_simple_control.util.get_model import get_model, get_heron_model
from types import SimpleNamespace
import time
import pickle
from importlib.resources import files


import pinocchio as pin
import numpy as np
import time
import argparse
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager

def plotEllipseMatplotlib(ax, opti_A, opti_c):
    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(opti_A)
    radii = 1.0 / np.sqrt(s)

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                np.dot([x[i, j], y[i, j], z[i, j]], rotation) + opti_c
            )

    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color="b", alpha=0.2)

# TODO: finish and verify
# make a separate function for a single ellipse based on the namespace
def visualizeEllipses(args, robot : RobotManager, ellipses):
    e, P = np.linalg.eig()
    ellipse_placement = pin.SE3(P, ellipse.center)
    #ellipse_placement = data.oMf[geom.parentFrame].act(geom.placement.act(ellipse_placement))
    ellipse_placement = data.oMf[geom.parentFrame].act(ellipse_placement)
    # TODO: add to MeshcatVisualizerWrapper,
    # add queue handling in visualize function
    viz.addEllipsoid(f"el_{ellipse.name}", sol_r, [0.3, 0.9, 0.3, 0.3])
    viz.applyConfiguration(f"el_{ellipse.name}", ellipse_placement)

# plotting the vertices in the ellipse in matplotlib for verification
def plotVertices(ax, vertices, nsubsample):
    """
    plot a 3xN array of vertices in 3d. If nsubsample is not 0, plot the main once in
    large red, the others in small green.
    """
    vert = vertices
    NS = nsubsample

    # Plot the vertices
    ax.plot3D(vert[::, 0], vert[::, 1], vert[::, 2], "g.", markersize=1)
    ax.plot3D(vert[::NS, 0], vert[::NS, 1], vert[::NS, 2], "r*")

    # Change the scalling for a regular one centered on the vertices.
    m, M = np.min(vert, 0), np.max(vert, 0)
    plot_center = (m + M) / 2
    plot_length = max(M - m) / 2
    ax.axes.set_xlim3d(
        left=plot_center[0] - plot_length, right=plot_center[0] + plot_length
    )
    ax.axes.set_ylim3d(
        bottom=plot_center[1] - plot_length, top=plot_center[1] + plot_length
    )
    ax.axes.set_zlim3d(
        bottom=plot_center[2] - plot_length, top=plot_center[2] + plot_length
    )

def visualizeVertices(args, robot : RobotManager):

    for i, geom in enumerate(robot.collision_model.geometryObjects):
        vertices = geom.geometry.vertices()
    
        for i in np.arange(0, vertices.shape[0]):
            viz.addSphere(f"world/point_{i}", 5e-3, [1, 0, 0, 0.8])
            vertix_pose = pin.SE3.Identity()
            vertix_pose.translation = vertices[i]
            #vertix_pose = data.oMi[geom.parentJoint].act(vertix_pose)
            vertix_pose = data.oMf[geom.parentFrame].act(geom.placement.act(vertix_pose))
            #viz.applyConfiguration(f"world/point_{i}", np.array(vertices[i].tolist() + [1, 0, 0, 0]))
            viz.applyConfiguration(f"world/point_{i}", vertix_pose)


# TODO: make this for every robot.
def computeEncapsulatingEllipses(args, robot : RobotManager):
    """
    computeEncapsulatingEllipses
    ----------------------------
    make convex approximations of the robot's links
    so that (self-) collision avoidace can be calculated more quickly
    and easily.
    this includes the need to group related links because otherwise
    we have multiple ellipses for the same part of the robot.
    the grouping of links has to be hardcoded and done manually
    and as it is non-trivial to make the grouping algorithmically,
    and since this is done once per robot there's no point
    to writing the algorithm (more work overall).
    """
    model, collision_model, visual_model, data = (robot.model, robot.collision_model, robot.visual_model, robot.data)
    viz = MeshcatVisualizer(model=model, collision_model=collision_model, visual_model=visual_model)
    #q = np.zeros(model.nq)
    q = pin.randomConfiguration(model)
    print(q)
    viz.display(q)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.updateGlobalPlacements(model,data)
    pin.computeAllTerms(model,data,q,np.zeros(model.nv))
    time.sleep(3)
    
    # vertex groups will have to be hardcoded per robot,
    # there's no getting around it
    vertices_grps = []
    vertices_grps_indeces = [
            #[0, 1], [2], [3], [4,5,6,7,8,9,10]  
            #[0, 1], [2], [3,4], [5,6,7,8,9,10]  
            [0, 1], [2], [3,4], [5,6,7,8,9,10]  
            ]
    vertices_grps_joint_parents = [collision_model.geometryObjects[0].parentJoint,
                            collision_model.geometryObjects[2].parentJoint,
                            collision_model.geometryObjects[3].parentJoint,
                            #collision_model.geometryObjects[4].parentJoint]
                            collision_model.geometryObjects[5].parentJoint]
    # index of parent joint needs to be here
    # we also maybe need parent frames
    #vertices_grps_joint_parents_indeces = [collision_model.geometryObjects[0].]
    # we'll np.vstack related vertices into vertices_grp
    for v in vertices_grps_indeces:
        # put them into their group
        vertices_grp = np.empty((0,3))
        for j in v:
            geom = collision_model.geometryObjects[j]
            # plot em in meshcat
            vertices = collision_model.geometryObjects[j].geometry.vertices()
            for i in np.arange(0, vertices.shape[0]):
                viz.addSphere(f"world/point_{i}", 5e-3, [1, 0, 0, 0.8])
                vertix_pose = pin.SE3.Identity()
                vertix_pose.translation = vertices[i]
                #vertix_pose = data.oMi[geom.parentJoint].act(vertix_pose)
                vertix_pose = data.oMf[geom.parentFrame].act(geom.placement.act(vertix_pose))
                #viz.applyConfiguration(f"world/point_{i}", np.array(vertices[i].tolist() + [1, 0, 0, 0]))
                viz.applyConfiguration(f"world/point_{i}", vertix_pose)

            vertices_in_joint_frame = []
            for g_v in vertices:
                # g_v is the vertex v expressed in the geometry frame.
                # Convert point from geometry frame to joint frame
                j_v = geom.placement.act(g_v)
                vertices_in_joint_frame.append(j_v)
            vertices_in_joint_frame = np.array(vertices_in_joint_frame)

            #vertices_grp = np.vstack((vertices_grp, collision_model.geometryObjects[j]))
            vertices_grp = np.vstack((vertices_grp, vertices_in_joint_frame))
        vertices_grps.append(vertices_grp)


    ellipses = []
    # go over grouped vertices and compute their ellipse fits
    #for i, geom in enumerate(collision_model.geometryObjects):
    #    vertices = geom.geometry.vertices()
    for i, vertices in enumerate(vertices_grps):
    
        cw = casadi.SX.sym("w", 3)
        exp = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
    
        """
        decide 
         - w in so3: ellipse orientation
         - r in r3: ellipse main dimensions
        minimizing:
          r1*r2*r3 the volum of the ellipse
        so that:
          r>=0
          for all points pk in a list, pk in ellipse
    
        """
        opti = casadi.Opti()
        var_w = opti.variable(3)
        var_r = opti.variable(3)
        var_c = opti.variable(3)
    
        # The ellipsoid matrix is represented by w=log3(R),diag(P) with R,P=eig(A)
        R = exp(var_w)
        A = R @ casadi.diag(1 / var_r**2) @ R.T
    
        totalcost = var_r[0] * var_r[1] * var_r[2]
        opti.subject_to(var_r >= 0)
#        for g_v in vertices:
        for j_v in vertices:
            # g_v is the vertex v expressed in the geometry frame.
            # Convert point from geometry frame to joint frame

#            print(j_v)
#            j_v = geom.placement.act(g_v)

            # Constraint the ellipsoid to be including the point
            opti.subject_to((j_v - var_c).T @ A @ (j_v - var_c) <= 1)
    
        ### SOLVE
        opti.minimize(totalcost)
        # remove prints i don't care about
        opts={}
        opts["verbose_init"] = False
        opts["verbose"] = False
        opts["print_time"] = False
        opts["ipopt.print_level"] = 0
        opti.solver("ipopt", opts)  # set numerical backend
        opti.set_initial(var_r, 10)
    
        sol = opti.solve_limited()
    
        sol_r = opti.value(var_r)
        sol_A = opti.value(A)
        sol_c = opti.value(var_c)
        sol_R = opti.value(exp(var_w))
    
        # TODO: add placement=pin.SE3(P, ellipse.center) and id=robot.model.getJointId(e.name)
        ellipse = SimpleNamespace(
            name=model.names[vertices_grps_joint_parents[i]],
            A=sol_A,
            center=sol_c)
        ellipses.append(ellipse)
        e, P = np.linalg.eig(sol_A)
        ellipse_placement = pin.SE3(P, ellipse.center)
        #ellipse_placement = data.oMf[geom.parentFrame].act(geom.placement.act(ellipse_placement))
        #ellipse_placement = data.oMf[geom.parentFrame].act(ellipse_placement)
        ellipse_placement = data.oMi[vertices_grps_joint_parents[i]].act(ellipse_placement)
        viz.addEllipsoid(f"el_{ellipse.name}", sol_r, [0.3, 0.9, 0.3, 0.3])
        viz.applyConfiguration(f"el_{ellipse.name}", ellipse_placement)
        print(ellipse)
    
    ellipses_path = files('ur_simple_control.robot_descriptions.ellipses').joinpath("ur5e_robotiq_ellipses.pickle")
    file = open(ellipses_path, 'wb')
    pickle.dump(ellipses, file)
    file.close()
    
        # Recover r,R from A (for fun)
    #    e, P = np.linalg.eig(sol_A)
    #    recons_r = 1 / e**0.5
    #    recons_R = P
    #
    #    # Build the ellipsoid 3d shape
    #    # Ellipsoid in meshcat
    #    viz.addEllipsoid("el", sol_r, [0.3, 0.9, 0.3, 0.3])
    #    # jMel is the placement of the ellipsoid in the joint frame
    #    jMel = pin.SE3(sol_R, sol_c)
    #
    #    # Place the body, the vertices and the ellispod at a random configuration oMj_rand
    #    oMj_rand = pin.SE3.Random()
    #    viz.applyConfiguration(viz.getViewerNodeName(geom, pin.VISUAL), oMj_rand)
    #    for i in np.arange(0, vertices.shape[0]):
    #        viz.applyConfiguration(
    #            f"world/point_{i}", oMj_rand.act(vertices[i]).tolist() + [1, 0, 0, 0]
    #        )
    #    viz.applyConfiguration("el", oMj_rand * jMel)
    #
    #    print(
    #        f'SimpleNamespace(name="{model.names[geom.parentJoint]}",\n'
    #        + f"                A=np.{repr(sol_A)},\n"
    #        + f"                center=np.{repr(sol_c)})"
    #    )
    #    time.sleep(5)
        # Matplotlib (for fun)
        #import matplotlib.pyplot as plt
        #
        #plt.ion()
        #from utils.plot_ellipse import plotEllipse, plotVertices
    
        #fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
        #plotEllipse(ax, sol_A, sol_c)
        #plotVertices(ax, np.vstack([geom.placement.act(p) for p in vertices]), 1)


def get_args():
    parser = getMinimalArgParser()
    parser.description = 'describe this example'
    # add more arguments here from different Simple Manipulator Control modules
    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = get_args()
    robot = RobotManager(args)

    computeEncapsulatingEllipses(args, robot)

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()
    #loop_manager.stopHandler(None, None)

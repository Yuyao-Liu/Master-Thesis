# override ugly tk widgets with system-based ttk ones
# change the import * thing if it's better for your ide
from tkinter import * 
from tkinter.ttk import *

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import pyautogui
from matplotlib.gridspec import GridSpec

from robot_stuff.InverseKinematics import InverseKinematicsEnv
from robot_stuff.drawing import *
from robot_stuff.inv_kinm import *
from make_run import makeRun

import numpy as np
import sys
# don't want to refactor yet, but obv
# TODO: refactor, have normal names for things
# it's just for getting the fps reading while optimizing
import time as ttime


DPI = 80
SLIDER_SIZE = 200
#DPI = 200

# TODO: call update_point function with the current slider value after updating 
# stuff like eliipse on/off so that you don't need to move the button to get it
# just call the slider string value to get the current slider value
# TODO: finish writing up reseting to the same starting position (joint values)
# TODO: readjust figure size so that it is not so javla small


#######################################################################
#                            CONTROL STUFF                            #
#######################################################################


def getController(controller_name):
    
    if controller_name == "invKinmQPSingAvoidE_kI":
        return invKinmQPSingAvoidE_kI
    if controller_name == "invKinm_Jac_T":
        return invKinm_Jac_T
    if controller_name == "invKinm_PseudoInv":
        return invKinm_PseudoInv
    if controller_name == "invKinm_dampedSquares":
        return invKinm_dampedSquares
    if controller_name == "invKinm_PseudoInv_half":
        return invKinm_PseudoInv_half
    if controller_name == "invKinmQP":
        return invKinmQP
    if controller_name == "invKinmQPSingAvoidE_kI":
        return invKinmQPSingAvoidE_kI
    if controller_name == "invKinmQPSingAvoidE_kM":
        return invKinmQPSingAvoidE_kM
    if controller_name == "invKinmQPSingAvoidManipMax":
        return invKinmQPSingAvoidManipMax

    return invKinm_dampedSquares


root = Tk()
root.wm_title("Embedding in Tk")

n_iters = 200
time = np.arange(n_iters)

# do the run
ik_env = InverseKinematicsEnv()
ik_env.robots.append(Robot_raw(robot_name="no_sim"))
ik_env.damping = 25
# putting it into this class so that python remembers it 'cos reasons, whatever
controller1 = getController("")
controller2 = getController("invKinm_Jac_T")
ik_env.data.append(makeRun(controller1, ik_env, n_iters, 0))
ik_env.data.append(makeRun(controller2, ik_env, n_iters, 1))

screensize = pyautogui.size()
SCREEN_WIDTH = screensize.width
SCREEN_HEIGHT = screensize.height
#SCALING_FACTOR_HEIGHT = 0.8
#SCALING_FACTOR_WIDTH = 0.5
SCALING_FACTOR_HEIGHT = 0.3
SCALING_FACTOR_WIDTH = 0.3

#######################################################################
#                             YARA'S PLOT                             #
#######################################################################

fig_ee = Figure(figsize=(SCREEN_WIDTH/DPI*SCALING_FACTOR_WIDTH, SCREEN_HEIGHT/DPI*SCALING_FACTOR_HEIGHT), dpi=DPI)
#exit_on_x(fig_ee)
#gs = GridSpec(3, 3, figure=fig_manip_graphs)

# Instantiate 3D reconstruction plot
#import scipy.io 
#data = scipy.io.loadmat('../geometry_matplotlib/data/orebro_castle')
##viewR = make_rotation_matrix3D(0, 0, 0)
#recX, recP = data['U'], data['P'][0]
#scale = recX.std() / 10
#cam_indices = np.arange(0,recP.shape[0],4) # np.random.permutation(recP.shape[0])[:200]
#rec3D_ax = fig.add_subplot(gs[:,0], projection='3d') 
rec3D_ax = fig_ee.add_subplot(projection='3d') 
rec3D_ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
#X_indices = np.arange(0,recX.shape[1],15) #np.random.permutation(recX.shape[1])
#num_pts_subset = 25
#for P_ in recP[cam_indices]:
#    camera_center_ = plot3D_camera(P_, ax=rec3D_ax, scale=scale, only_center=True)
#    camera_center_.set(markersize=1)
#camera_center, camera_coordinate_system = plot3D_camera(recP[0], ax=rec3D_ax, scale=scale)
#rec_points3D = plot3D(recX[:,X_indices[:num_pts_subset]], ax=rec3D_ax, title='')[0][0]
#set_equal_axis3D(rec3D_ax)
#rec3D_ax.view_init(elev=0, azim=0, roll=0)
#rec3D_ax.set_title('3D Reconstruction')
###########################################################################

#######################################################################
#                          MANIPULATOR PLOT                           #
#######################################################################

# robot plot
#ik_env.ax = fig.add_subplot(132, projection='3d') 
#ik_env.ax = fig.add_subplot(gs[:,1], projection='3d') 
fig_manipulator = Figure(figsize=(SCREEN_WIDTH/DPI*SCALING_FACTOR_WIDTH, SCREEN_HEIGHT/DPI*SCALING_FACTOR_HEIGHT), dpi=DPI)
ik_env.ax = fig_manipulator.add_subplot(projection='3d') 
ik_env.ax.plot(np.array([0]), np.array([0]), np.array([1.5]), c='b')
ik_env.ax.plot(np.array([0]), np.array([0]), np.array([-1.5]), c='b')
ik_env.ax.set_xlim([-1.0,1.0])
ik_env.ax.set_ylim([-1.0,1.0])
link_colors = ['black', 'violet']
trajectory_plots = []
for robot_index, robot in enumerate(ik_env.robots):
    robot.initDrawing(ik_env.ax, link_colors[robot_index])
    # ee point
    ik_env.p_e_point_plots.append(drawPoint(ik_env.ax, ik_env.data[robot_index]['p_es'][0], 'red', 'o'))
    # let's add the trajectory to this
    trajectory_plot, = ik_env.ax.plot(ik_env.data[robot_index]['p_es'][:,0], ik_env.data[robot_index]['p_es'][:,1], ik_env.data[robot_index]['p_es'][:,2], color='blue')
    trajectory_plots.append(trajectory_plot)
# goal point
ik_env.goal_point_plot = drawPoint(ik_env.ax, ik_env.goal, 'maroon', '*')
#background_manipulator = fig_manipulator.canvas.copy_from_bbox(fig_manipulator.bbox)


# let's add the manipulability ellipse
for robot_index, robot in enumerate(ik_env.robots):
    radii = 1.0/ik_env.data[robot_index]["manip_ell_eigenvals"][0] * 0.1
    u = np.linspace(0.0, 2.0 * np.pi, 60)     
    v = np.linspace(0.0, np.pi, 60)     
    x = radii[0] * np.outer(np.cos(u), np.sin(v))     
    y = radii[1] * np.outer(np.sin(u), np.sin(v))     
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):         
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], ik_env.data[robot_index]["manip_elip_svd_rots"][0]) + ik_env.data[robot_index]['p_es'][0]

    ik_env.ellipsoid_surf_plots.append(ik_env.ax.plot_surface(x, y, z,  rstride=3, cstride=3,
                    color='pink', linewidth=0.1,
                    alpha=0.3, shade=True))

#######################################################################
#                        manipulability plots                         #
#######################################################################

# manipulability ellipsoid eigenvalues
fig_manip_graphs = Figure(figsize=(SCREEN_WIDTH/DPI*SCALING_FACTOR_WIDTH, SCREEN_HEIGHT/DPI*SCALING_FACTOR_HEIGHT), dpi=DPI)
#ax = fig_manip_graphs.add_subplot(gs[0,-1])
ax = fig_manip_graphs.add_subplot(311)
ax.set_title('Manipulability ellipsoid eigenvalues')
ax.grid()
ax.set_xticks([])
eigen1_lines = []
eigen2_lines = []
eigen3_lines = []
for robot_index, robot in enumerate(ik_env.robots):
    eigen1_line, = ax.plot(time, ik_env.data[robot_index]["manip_ell_eigenvals"][:,0], color=link_colors[robot_index])
    eigen1_lines.append(eigen1_line)
    eigen2_line, = ax.plot(time, ik_env.data[robot_index]["manip_ell_eigenvals"][:,1], color=link_colors[robot_index])
    eigen2_lines.append(eigen2_line)
    eigen3_line, = ax.plot(time, ik_env.data[robot_index]["manip_ell_eigenvals"][:,2], color=link_colors[robot_index])
    eigen3_lines.append(eigen3_line)
point_in_time_eigen1_line = ax.axvline(x=0, color='red')

# manipulability index (volume of manipulability ellipsoid)
#ax = fig.add_subplot(gs[1,-1]) 
ax = fig_manip_graphs.add_subplot(312) 
ax.set_title('Manipulability index')
ax.grid()
ax.set_xticks([])
manip_index_lines = []
for robot_index, robot in enumerate(ik_env.robots):
    manip_index_line, = ax.plot(time, ik_env.data[robot_index]['manip_indeces'], color=link_colors[robot_index])
    manip_index_lines.append(manip_index_line)
point_in_time_manip_index_line = ax.axvline(x=0, color='red')

# dist to goal (this could be/should be elsewhere)
#ax = fig.add_subplot(gs[2,-1]) 
ax = fig_manip_graphs.add_subplot(313) 
ax.set_title('Distance to goal')
ax.grid()
ax.set_xlabel('iter')
dist_to_goal_lines = []
for robot_index, robot in enumerate(ik_env.robots):
    dist_to_goal_line, = ax.plot(time, ik_env.data[robot_index]['dists_to_goal'], color=link_colors[robot_index])
    dist_to_goal_lines.append(dist_to_goal_line)
point_in_time_dist_to_goal_line = ax.axvline(x=0, color='red')


#######################################################################
#                              IMU plots                              #
#######################################################################

fig_imu = Figure(figsize=(SCREEN_WIDTH/DPI*SCALING_FACTOR_WIDTH, SCREEN_HEIGHT/DPI*SCALING_FACTOR_HEIGHT), dpi=DPI)
#ax = fig_manip_graphs.add_subplot(gs[0,-1])
v_x_lines = []
v_y_lines = []
v_z_lines = []
omega_x_lines = []
omega_y_lines = []
omega_z_lines = []

ax_v_x = fig_imu.add_subplot(321)
ax_v_x.grid()
ax_v_y = fig_imu.add_subplot(322)
ax_v_y.grid()
ax_v_z = fig_imu.add_subplot(323)
ax_v_z.grid()
ax_omega_x = fig_imu.add_subplot(324)
ax_omega_x.grid()
ax_omega_y = fig_imu.add_subplot(325)
ax_omega_y.grid()
ax_omega_z = fig_imu.add_subplot(326)
ax_omega_z.grid()
ax_v_x.set_title('Linear velocity x')
ax_v_y.set_title('Linear velocity y')
ax_v_z.set_title('Linear velocity z')
ax_omega_x.set_title('Angular velocity x')
ax_omega_y.set_title('Angular velocity y')
ax_omega_z.set_title('Angular velocity z')
ax_v_x.set_xticks([])
ax_v_y.set_xticks([])
ax_v_z.set_xticks([])
ax_omega_x.set_xticks([])
ax_omega_y.set_xticks([])
ax_omega_z.set_xticks([])
for robot_index, robot in enumerate(ik_env.robots):
    v_x_line, = ax_v_x.plot(time, ik_env.data[robot_index]["vs"][:,0], color=link_colors[robot_index])
    v_y_line, = ax_v_y.plot(time, ik_env.data[robot_index]["vs"][:,1], color=link_colors[robot_index])
    v_z_line, = ax_v_z.plot(time, ik_env.data[robot_index]["vs"][:,2], color=link_colors[robot_index])
    omega_x_line, = ax_omega_x.plot(time, ik_env.data[robot_index]["vs"][:,3], color=link_colors[robot_index])
    omega_y_line, = ax_omega_y.plot(time, ik_env.data[robot_index]["vs"][:,4], color=link_colors[robot_index])
    omega_z_line, = ax_omega_z.plot(time, ik_env.data[robot_index]["vs"][:,5], color=link_colors[robot_index])
    v_x_lines.append(v_x_line)
    v_y_lines.append(v_y_line)
    v_z_lines.append(v_z_line)
    omega_x_lines.append(omega_x_line)
    omega_y_lines.append(omega_y_line)
    omega_z_lines.append(omega_z_line)

point_in_time_ax_v_x_line = ax_v_x.axvline(x=0, color='red')
point_in_time_ax_v_y_line = ax_v_y.axvline(x=0, color='red')
point_in_time_ax_v_z_line = ax_v_z.axvline(x=0, color='red')
point_in_time_ax_omega_x_line = ax_omega_x.axvline(x=0, color='red')
point_in_time_ax_omega_y_line = ax_omega_y.axvline(x=0, color='red')
point_in_time_ax_omega_z_line = ax_omega_z.axvline(x=0, color='red')


#######################################################################
#           putting plots into: frames -> notebooks -> tabs           #
#######################################################################
# adding whatever into this
notebook_left = Notebook(root, height=int(SCREEN_HEIGHT))#, width=700)
notebook_right = Notebook(root, height=int(SCREEN_HEIGHT))#, width=700)
frame_manipulator = Frame(notebook_left)#, width=400, height=400)
frame_manip_graphs = Frame(notebook_right)#, width=400, height=400)
frame_imu = Frame(notebook_right)#, width=400, height=400)
frame_ee = Frame(notebook_left)#, width=400, height=400)
tabs_left = notebook_left.add(frame_manipulator, text='manipulator')
tabs_left = notebook_left.add(frame_ee, text="end-effector")
tabs_right = notebook_right.add(frame_manip_graphs, text='manipulator-graphs')
tabs_right = notebook_right.add(frame_imu, text="ee-graphs")
#frame_manipulator.pack(fill='both', expand=True)
#frame_manip_graphs.pack(fill='both', expand=True)
notebook_left.grid(row=0, column=0, sticky='ew')
notebook_right.grid(row=0, column=1, sticky='ew')
#notebook_left.pack(fill='both', expand=True)
#canvas = FigureCanvasTkAgg(fig, master=root) 

# tkinterize these plots
canvas_manipulator = FigureCanvasTkAgg(fig_manipulator, master=frame_manipulator) 
canvas_manipulator.draw()
# NEW
# TODO maybe you want another background idk
# worked elsewhere with figure.bbox
background_manipulator = canvas_manipulator.copy_from_bbox(fig_manipulator.bbox)
#background_manipulator = canvas_manipulator.copy_from_bbox(canvas_manipulator.bbox)

canvas_manipulator_widget = canvas_manipulator.get_tk_widget()     
canvas_manipulator_widget.grid(row=0, column=0) 
canvas_manipulator._tkcanvas.grid(row=1, column=0)   
canvas_manipulator.draw()

canvas_ee = FigureCanvasTkAgg(fig_ee, master=frame_ee) 
canvas_ee_widget = canvas_ee.get_tk_widget()     
canvas_ee_widget.grid(row=0, column=0) 
canvas_ee._tkcanvas.grid(row=1, column=0)   
canvas_ee.draw()

canvas_manip_graphs = FigureCanvasTkAgg(fig_manip_graphs, master=frame_manip_graphs) 
canvas_manip_graphs_widget = canvas_manip_graphs.get_tk_widget()     
canvas_manip_graphs_widget.grid(row=0, column=0) 
canvas_manip_graphs._tkcanvas.grid(row=1, column=0)   
canvas_manip_graphs.draw()

canvas_imu = FigureCanvasTkAgg(fig_imu, master=frame_imu) 
canvas_imu_widget = canvas_manip_graphs.get_tk_widget()     
canvas_imu_widget.grid(row=0, column=0) 
canvas_imu._tkcanvas.grid(row=1, column=0)   
canvas_imu.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar_manipulator = NavigationToolbar2Tk(canvas_manipulator, frame_manipulator, pack_toolbar=False)
toolbar_manipulator.update()
toolbar_manipulator.grid(column=0, row=2)
toolbar_manip_graphs = NavigationToolbar2Tk(canvas_manip_graphs, frame_manip_graphs, pack_toolbar=False)
toolbar_manip_graphs.update()
toolbar_manip_graphs.grid(column=0, row=2)
toolbar_ee = NavigationToolbar2Tk(canvas_ee, frame_ee, pack_toolbar=False)
toolbar_ee.update()
toolbar_ee.grid(column=0, row=2)
toolbar_imu = NavigationToolbar2Tk(canvas_imu, frame_imu, pack_toolbar=False)
toolbar_imu.update()
toolbar_imu.grid(column=0, row=2)

#######################################################################
#                 functions for widgets to control the plots          #
#######################################################################

# keyboard inputs
# whatever, doesn't hurt, could be used
#canvas.mpl_connect(
#    "key_press_event", lambda event: print(f"you pressed {event.key}"))
#canvas.mpl_connect("key_press_event", key_press_handler)

# new_val is what the widget gives you
# you define what needs to happen to plots in Figure below,
# and call canvas.draw
def update_points(new_val):
# ee plot
###########################################################################
    start = ttime.time()
    #fig_manipulator.canvas.restore_region(background_manipulator)
    canvas_manipulator.restore_region(background_manipulator)
    index = int(np.floor(float(new_val)))
    
    # Update 3D reconstruction plot
    # plot3D_camera(recP[index], scale=scale)
#    C, Q = camera_data_for_plot3D(recP[cam_indices[index]])
#    camera_center.set_data_3d(*C)
#    C = C.flatten()
#    for ii in range(len(camera_coordinate_system)):
#        camera_coordinate_system[ii].set_data([C[0], C[0] + scale * Q[0,ii]],
#                                              [C[1], C[1] + scale * Q[1,ii]])
#        camera_coordinate_system[ii].set_3d_properties([C[2], C[2] + scale * Q[2,ii]])
#    recX_ = recX[:,X_indices[:num_pts_subset*(index+1)]]
#    rec_points3D.set_data_3d(recX_[0,:], recX_[1,:], recX_[2,:])
###########################################################################

    ik_env.ax.set_title(str(index) + 'th iteration toward goal')
    # animate 3d manipulator
    for robot_index, robot in enumerate(ik_env.robots):
        ik_env.robots[robot_index].setJoints(ik_env.data[robot_index]["qs"][index])
        ik_env.robots[robot_index].drawStateAnim()

    # NEW AND BROKEN
        for link in robot.lines:
            for line in link:
                ik_env.ax.draw_artist(line)
        # all these are in lists as that's what the line plot wants, 
        # despite the fact that we have single points
        point_in_time_eigen1_line.set_xdata([time[index]])
        point_in_time_manip_index_line.set_xdata([time[index]])
        point_in_time_dist_to_goal_line.set_xdata([time[index]])
        point_in_time_ax_v_x_line.set_xdata([time[index]])
        point_in_time_ax_v_y_line.set_xdata([time[index]])
        point_in_time_ax_v_z_line.set_xdata([time[index]])
        point_in_time_ax_omega_x_line.set_xdata([time[index]])
        point_in_time_ax_omega_y_line.set_xdata([time[index]])
        point_in_time_ax_omega_z_line.set_xdata([time[index]])

        # ellipsoid update
        radii = 1.0/ik_env.data[robot_index]["manip_ell_eigenvals"][index] * 0.1
        u = np.linspace(0.0, 2.0 * np.pi, 60)     
        v = np.linspace(0.0, np.pi, 60)     
        x = radii[0] * np.outer(np.cos(u), np.sin(v))     
        y = radii[1] * np.outer(np.sin(u), np.sin(v))     
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):         
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], ik_env.data[robot_index]["manip_elip_svd_rots"][index]) + ik_env.data[robot_index]['p_es'][index]

        if ellipse_on_off_var.get():
            try:
                ik_env.ellipsoid_surf_plots[robot_index].remove()
            except ValueError:
                pass
            ik_env.ellipsoid_surf_plots[robot_index] = ik_env.ax.plot_surface(x, y, z,  rstride=3, cstride=3,
                            color='pink', linewidth=0.1,
                            alpha=0.3, shade=True) 

        ik_env.p_e_point_plots[robot_index].set_data([ik_env.data[robot_index]['p_es'][index][0]], [ik_env.data[robot_index]['p_es'][index][1]])
        ik_env.p_e_point_plots[robot_index].set_3d_properties([ik_env.data[robot_index]['p_es'][index][2]])
    canvas_ee.draw()
    # NEW AND BROKEN
#    fig_manipulator.canvas.blit(fig_manipulator.bbox)
#    fig_manipulator.canvas.flush_events()
    canvas_manipulator.blit(fig_manipulator.bbox)
    canvas_manipulator.flush_events()
    #canvas_manipulator.draw()
    # might need to manually update all artists here from ax_something
    #canvas_manipulator.blit(fig_manipulator.bbox)
    canvas_manip_graphs.draw()
    canvas_imu.draw()
    # TODO update may not be needed as we're going by slider here
    root.update()
    end = ttime.time()
    print("time per update:", end - start)
    print("fps", 1 / (end - start))


def update_goal_x(new_val):
    goal_x = float(new_val)
    ik_env.goal[0] = goal_x
    ik_env.goal_point_plot.set_data([ik_env.goal[0]], [ik_env.goal[1]])
    ik_env.goal_point_plot.set_3d_properties([ik_env.goal[2]])
    canvas_ee.draw()
    canvas_manipulator.draw()
    canvas_manip_graphs.draw()
    # TODO update may not be needed as we're going by slider here
    root.update()

def update_goal_y(new_val):
    goal_y = float(new_val)
    ik_env.goal[1] = goal_y
    ik_env.goal_point_plot.set_data([ik_env.goal[0]], [ik_env.goal[1]])
    ik_env.goal_point_plot.set_3d_properties([ik_env.goal[2]])
    canvas_ee.draw()
    canvas_manipulator.draw()
    canvas_manip_graphs.draw()
    # TODO update may not be needed as we're going by slider here
    root.update()

def update_goal_z(new_val):
    goal_z = float(new_val)
    ik_env.goal[2] = goal_z
    ik_env.goal_point_plot.set_data([ik_env.goal[0]], [ik_env.goal[1]])
    ik_env.goal_point_plot.set_3d_properties([ik_env.goal[2]])
    canvas_ee.draw()
    canvas_manipulator.draw()
    canvas_manip_graphs.draw()
    # TODO update may not be needed as we're going by slider here
    root.update()



def reset():
    ik_env.reset()
#    ik_env.goal_point_plot.remove()
#    ik_env.goal_point_plot = drawPoint(ik_env.ax, ik_env.goal, 'red', 'o')
#    print(controller_string1.get())
#    print(controller_string2.get())
    controller1 = getController(controller_string1.get())
    controller2 = getController(controller_string2.get())
    controllers = [controller1, controller2]
    for robot_index, robot in enumerate(ik_env.robots):
        ik_env.data.append(makeRun(controllers[robot_index], ik_env, n_iters, robot_index))
        trajectory_plots[robot_index].set_data(ik_env.data[robot_index]['p_es'][:,0], ik_env.data[robot_index]['p_es'][:,1])
        trajectory_plots[robot_index].set_3d_properties(ik_env.data[robot_index]['p_es'][:,2])
        eigen1_lines[robot_index].set_ydata(ik_env.data[robot_index]["manip_ell_eigenvals"][:,0])
        eigen2_lines[robot_index].set_ydata(ik_env.data[robot_index]["manip_ell_eigenvals"][:,1])
        eigen3_lines[robot_index].set_ydata(ik_env.data[robot_index]["manip_ell_eigenvals"][:,2])
        manip_index_lines[robot_index].set_ydata(ik_env.data[robot_index]['manip_indeces'])
        dist_to_goal_lines[robot_index].set_ydata(ik_env.data[robot_index]['dists_to_goal'])
    update_points(0)
    canvas_ee.draw()
    canvas_manipulator.draw()
    canvas_manip_graphs.draw()
    root.update()

def play():
    pass

# ellipse on/off
def add_remove_ellipse():
    try:
        for robot_index, robot in enumerate(ik_env.robots):
            ik_env.ellipsoid_surf_plots[robot_index].remove()
    except ValueError:
        pass


# NOTE: this thing is not used
same_starting_position_on_off_var = IntVar()   
same_starting_position_on_off_var.set(0)
same_starting_position_checkbutton= Checkbutton(root, text = "same starting position on/off",
                      variable = same_starting_position_on_off_var,
                      onvalue = 1,
                      offvalue = 0,
#                      command=same_starting_position_cmd,
                      )


#######################################################################
#                     PLACING ELEMENTS IN THE GUI                     #
#######################################################################

# LEFT PANE BELOW MANIP/EE PLOTS
frame_below_manipulator_plot = Frame(frame_manipulator)
frame_below_manipulator_plot.grid(column=0, row=3)

# set controller 1
frame_controller_menu1 = Frame(frame_below_manipulator_plot)
controller_string1 = StringVar(frame_controller_menu1) 
controller_string1.set("invKinm_dampedSquares") # default value 
controller_menu1 = OptionMenu(frame_controller_menu1, controller_string1, 
                "invKinmQPSingAvoidE_kI",
                "invKinm_Jac_T",
                "invKinm_PseudoInv",
                "invKinm_dampedSquares",
                "invKinm_PseudoInv_half",
                "invKinmQP",
                "invKinmQPSingAvoidE_kI",
                "invKinmQPSingAvoidE_kM",
                "invKinmQPSingAvoidManipMax",
               ) 
controller_string1.set("invKinm_dampedSquares")
controller_menu1.grid(column=1, row=0)
Label(frame_controller_menu1, text="Robot 1 controller:", background='yellow').grid(row=0, column=0, pady=4, padx = 4)
frame_controller_menu1.grid(column=0, row=0)

# set controller 2
frame_controller_menu2 = Frame(frame_below_manipulator_plot)
controller_string2 = StringVar(frame_controller_menu2) 
controller_menu2 = OptionMenu(frame_controller_menu2, controller_string2, 
                "invKinmQPSingAvoidE_kI",
                "invKinm_Jac_T",
                "invKinm_PseudoInv",
                "invKinm_dampedSquares",
                "invKinm_PseudoInv_half",
                "invKinmQP",
                "invKinmQPSingAvoidE_kI",
                "invKinmQPSingAvoidE_kM",
                "invKinmQPSingAvoidManipMax",
               ) 
controller_string2.set("invKinm_Jac_T") # default value 
Label(frame_controller_menu1, text="Robot 1 controller:", background='black', foreground='white').grid(row=0, column=0, pady=4, padx = 4)
controller_menu2.grid(column=1, row=0)
Label(frame_controller_menu2, text="Robot 2 controller:", background='#EE82EE').grid(row=0, column=0, pady=4, padx = 4)
frame_controller_menu2.grid(column=0, row=1)


ellipse_on_off_var = IntVar()   
ellipse_on_off_var.set(1)

ellipse_checkbutton= Checkbutton(frame_below_manipulator_plot, text = "ellipse on/off",
                      variable = ellipse_on_off_var,
                      onvalue = 1,
                      offvalue = 0,
                      command=add_remove_ellipse,
                      )
ellipse_checkbutton.grid(column=1, row=0)


frame_goal_x = Frame(frame_below_manipulator_plot)
Label(frame_goal_x, text="goal x").grid(row=0, column=0, pady=4, padx = 4)
slider_goal_x = Scale(frame_goal_x, from_=-1.0, to=1.0, length=SLIDER_SIZE, orient=HORIZONTAL,
                              command=update_goal_x)
slider_goal_x.grid(column=1, row=0)
frame_goal_x.grid(column=1, row=1)

frame_goal_y = Frame(frame_below_manipulator_plot)
Label(frame_goal_y, text="goal y").grid(row=0, column=0, pady=4, padx = 4)
slider_goal_y = Scale(frame_goal_y, from_=-1.0, to=1.0, length=SLIDER_SIZE, orient=HORIZONTAL,
                              command=update_goal_y)
slider_goal_y.grid(column=1, row=0)
frame_goal_y.grid(column=1, row=2)


frame_goal_z = Frame(frame_below_manipulator_plot)
Label(frame_goal_z, text="goal z").grid(row=0, column=0, pady=4, padx = 4)
slider_goal_z = Scale(frame_goal_z, from_=-1.0, to=1.0, length=SLIDER_SIZE, orient=HORIZONTAL,
                              command=update_goal_z)
slider_goal_z.grid(column=1, row=0)
frame_goal_z.grid(column=1, row=3)


frame_update = Frame(frame_manip_graphs)
Label(frame_update, text="Time").grid(row=0, column=0, pady=4, padx = 4)
slider_update = Scale(frame_update, from_=0, to=n_iters - 1, length=SLIDER_SIZE, orient=HORIZONTAL,
                              command=update_points)#, label="Frequency [Hz]")
slider_update.grid(column=1, row=0)
frame_update.grid(column=0, row=3)

frame_update2 = Frame(frame_imu)
Label(frame_update2, text="Time").grid(row=0, column=0, pady=4, padx = 4)
slider_update = Scale(frame_update2, from_=0, to=n_iters - 1, length=SLIDER_SIZE, orient=HORIZONTAL,
                              command=update_points)#, label="Frequency [Hz]")
slider_update.grid(column=1, row=0)
frame_update2.grid(column=0, row=3)


button_quit = Button(master=frame_manip_graphs, text="Quit", command=root.destroy)
button_reset = Button(master=frame_manip_graphs, text="New run", command=reset)
button_reset.grid(column=0, row=4)
button_quit.grid(column=0, row=5)

button_quit2 = Button(master=frame_imu, text="Quit", command=root.destroy)
button_reset2 = Button(master=frame_imu, text="New run", command=reset)
button_reset2.grid(column=0, row=4)
button_quit2.grid(column=0, row=5)

update_points(0)
slider_goal_x.set(ik_env.goal[0])
slider_goal_y.set(ik_env.goal[1])
slider_goal_z.set(ik_env.goal[2])
mainloop()

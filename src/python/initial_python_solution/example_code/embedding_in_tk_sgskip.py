#import tkinter
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

from robot_stuff.InverseKinematics import InverseKinematicsEnv
from robot_stuff.drawing import *
from robot_stuff.inv_kinm import *

import numpy as np


def policy(robot, desired_goal):
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinm_Jac_T(robot, desired_goal)
#    del_thet = invKinm_PseudoInv(robot, desired_goal)
#    del_thet = invKinm_dampedSquares(robot, desired_goal)
    del_thet = invKinm_PseudoInv_half(robot, desired_goal)
#    del_thet = invKinmQP(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kM(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidManipMax(robot, desired_goal)

    return del_thet

# TODO move all tkinter stuff to one file,
# and all plot stuff to one file to avoid this schizo mess
root = Tk()
root.wm_title("Embedding in Tk")

# mock data 'cos i was lazy with cleaning the rest
start = 0
end = 10
n_pts = 2000
dt = (end - start) / n_pts
time = np.linspace(start,end,n_pts)

angular_v_x = np.sin(time/2)
angular_v_y = np.cos(time/3)
angular_v_z = np.sin(time/5)
v_x = time / 2
v_y = time **2 / (time + 2)
v_z = np.cos(time / 10)

# integrate velocities to get positions
angular_p_x = np.cumsum(angular_v_x * dt)
angular_p_y = np.cumsum(angular_v_y * dt)
angular_p_z = np.cumsum(angular_v_z * dt)
p_x = np.cumsum(v_x * dt)
p_y = np.cumsum(v_y * dt)
p_z = np.cumsum(v_z * dt)


# when you plot something on the axis, it returns you the object of that plot
# you then update this later on
# this is much much faster than replotting everything every time (even without blitting,
# which can be implemented if necessary)

# the functions to update these plot objects can be found on matplotlib pages by just googling



# the whole figure
fig = Figure()

# init robot stuff
ik_env = InverseKinematicsEnv()
# no way this flies, but we have to try
ik_env.ax = fig.add_subplot(121, projection='3d') 
# these are for axes scaling which does not happen automatically
ik_env.ax.plot(np.array([0]), np.array([0]), np.array([1.5]), c='b')
ik_env.ax.plot(np.array([0]), np.array([0]), np.array([-1.5]), c='b')
# NOTE hope this applies just to this ax, otherwise change to ax.xlim or whatever
ik_env.ax.set_xlim([-1.5,1.5])
ik_env.ax.set_ylim([-0.5,0.5])
color_link = 'black'
ik_env.robot.initDrawing(ik_env.ax, color_link)

# do the run


# TODO may use this as trajectory later?
#position_line, = ax.plot(p_x, p_y, p_z, color='blue')
## the quive sucks for this purpose as the tangets stick to the curve and you don't see anything
##mask = np.arange(n_pts) % 200 == 0
##ax.quiver(p_x[mask], p_y[mask], p_z[mask], v_x[mask], v_y[mask], v_z[mask], color='orange')
## there's no colormap over a line, so just plot colored line segments (many lines)
#for i in range(n_pts - 1):
#    ax.plot(p_x[i:i+2], p_y[i:i+2], p_z[i:i+2], color=plt.cm.plasma((i)/n_pts))
#fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap=plt.cm.plasma), orientation='horizontal')
#
## needs to be a line 'cos that's what we can update
#point_in_time_position_line, = ax.plot(np.zeros(1), np.zeros(1),np.zeros(1),
#                                       marker='.', markersize=12, linestyle='None', color='red')



# whatever side bullshit
ax = fig.add_subplot(322) 
ax.set_xlabel('t/s')
ang_v_x_line, = ax.plot(time, angular_v_x)
point_in_time_angular_v_x_line, = ax.plot(np.zeros(1), np.zeros(1), 
                                          marker='.', markersize=12, linestyle='None', color='red')

ax = fig.add_subplot(324) 
ax.set_xlabel('t/s')
ang_v_z_line, = ax.plot(time, angular_v_y)
point_in_time_angular_v_y_line, = ax.plot(np.zeros(1), np.zeros(1), 
                                          marker='.', markersize=12, linestyle='None', color='red')

ax = fig.add_subplot(326) 
ax.set_xlabel('t/s')
ang_v_y_line, = ax.plot(time, angular_v_z)
point_in_time_angular_v_z_line, = ax.plot(np.zeros(1), np.zeros(1), 
                                          marker='.', markersize=12, linestyle='None', color='red')





canvas = FigureCanvasTkAgg(fig, master=root) 
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = Button(master=root, text="Quit", command=root.destroy)


# new_val is what the widget gives you
# you define what needs to happen to plots in Figure below,
# and call canvas.draw
def update_point(new_val):
    new_val = int(np.floor(float(new_val)))
    
    # all these are in lists as that's what the line plot wants, 
    # but we use single points
#    point_in_time_position_line.set_data_3d([p_x[new_val]], [p_y[new_val]],  [p_z[new_val]])
    point_in_time_angular_v_x_line.set_data([time[new_val]], [angular_v_x[new_val]])
    point_in_time_angular_v_y_line.set_data([time[new_val]], [angular_v_y[new_val]])
    point_in_time_angular_v_z_line.set_data([time[new_val]], [angular_v_z[new_val]])

    # required to update canvas and attached toolbar!
    canvas.draw()

slider_update = Scale(root, from_=0, to=n_pts - 1, length=500, orient=HORIZONTAL,
                              command=update_point)#, label="Frequency [Hz]")

# TODO no way in hell this flies, but we have to try
# does not update main frame until it finishes
def updateIKAnim():
    # do the step
    for i in range(50):
        action = policy(ik_env.robot, ik_env.goal)
        action_fix = list(action)
        action_fix.append(1.0)
        action_fix = np.array(action_fix)
        obs, reward, done, info = ik_env.step(action_fix)
        # animate
        ik_env.robot.drawStateAnim()
        ik_env.ax.set_title(str(ik_env.n_of_tries_for_point) + 'th iteration toward goal')
        #ik_env.ax.set_title(str(np.random.random()) + 'th iteration toward goal')
        drawPoint(ik_env.ax, ik_env.goal, 'red')
        canvas.draw()
        root.update()

button_anim = Button(master=root, text="anim_step", command=updateIKAnim)



# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
# NOTE: imo it's fine to just pack now and work on the layout later once
# everything we want to plot is known
button_anim.pack(side=BOTTOM)
button_quit.pack(side=BOTTOM)
slider_update.pack(side=BOTTOM)
toolbar.pack(side=BOTTOM, fill=X)
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

mainloop()

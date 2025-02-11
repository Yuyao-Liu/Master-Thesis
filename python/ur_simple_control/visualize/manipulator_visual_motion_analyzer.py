# override ugly tk widgets with system-based ttk ones
# change the import * thing if it's better for your ide
# TODO: remove last thing after you switch ellipse off
# TODO: set reasonable y-axis limits for side plots
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
from ur_simple_control.visualize.make_run import makeRun, loadRun
from ur_simple_control.managers import RobotManager
from ur_simple_control.util.logging_utils import loadRunForAnalysis, cleanUpRun

import numpy as np
# it is the best solution for a particular problem
from collections import namedtuple
# needed to communicate with the gui in real time
from multiprocessing import Queue
# for local thread which manages local queue (can be made better,
# but again, who cares, all of this is ugly as hell anyway ('cos it's front-end))
import threading
import time 
import pickle



# TODO: call update_point function with the current slider value after updating 
# stuff like elipse on/off so that you don't need to move the button to get it
# just call the slider string value to get the current slider value
# TODO: finish writing up reseting to the same starting position (joint values)


#######################################################################
#                            CONTROL STUFF                            #
#######################################################################

# TODO: remove this/import it from clik
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

class GetCommandThread(threading.Thread):
    """
    GetCommandThread
    -----------------
    - NOTE: NOT USED ATM
    - requires separate thread to accomodate the updating
      weirdness of tkinter. 
    - i know how this works, so i'm running with it,
      if you want to make it better, be my guest, code it up
    - TODO: make it work for the new application
    --> just send q commands here
    - there are 2 queues because i didn't know what i was doing honestly,
      but hey, it works and it ain't hurting nobody
    - the point is that you generate an event for the gui, and then get what
      that event actually was by poping from the queue.
      and you can shove anything, ex. whole dicts into the queue because python
    """
    def __init__(self, queue, localQueue, _check):
        super(GetCommandThread, self).__init__()
        self.queue = queue
        self.localQueue = localQueue
        self._check = _check

    def run(self):
        #print("worker thread doing stuff")
        commands = self.queue.get()
        # maybe just wait untill it's non-empty with another blocking call?
        # then .get() from the main thread
        self.localQueue.put(commands)
#        if random.randint(0, 1) == 0:
#            self.localQueue.put({'set_image': 'smile', 'set_text': "ready to work" })
#        else:
#            self.localQueue.put({'set_image': 'thumb_up', 'set_text': "updated" })

        if self._check.get() == 0:
            self._check.set(1)
        #    print('registered new callback')
        else:
            self._check.set(0)
        #    print('registered new callback')


# shove artists into dicts, not lists,
# so that they have names. the for loop functions the same way,
# but you get to know what you have, which might be useful later.
# and then update all of them in a single for loop.
# shove all other non-updating artists into a single list.
# this list is to be updated only if you load a new run.
# or even skip this and reload everything, who cares.
class ManipulatorVisualMotionAnalyzer:
    """
    ManipulatorVisualMotionAnalyzer
    ----------------------------------
    - for now leaving run generation here for easier testing
    - later load run and run on that
    - add option to load run while the robot is running
    - some possibly unused stuff will be added here as a weird transplant
      from an old project, but will be trimmed later
    """
    def __init__(self, root, queue, data, real_time_flag, **kwargs):
        # need to put this in the main program, 
        # so you need to pass it here to use it
        self.root = root
        self.real_time_flag = real_time_flag
        self.root.wm_title("Embedding in Tk")
        # for real-time updates
        self.queue = queue
        #######################
        #  visual parameters  #
        #######################
        self.DPI = 80
        self.SLIDER_SIZE = 200
        #DPI = 200
        screensize = pyautogui.size()
        self.SCREEN_WIDTH = screensize.width
        self.SCREEN_HEIGHT = screensize.height
        #self.SCALING_FACTOR_HEIGHT = 0.8
        #self.SCALING_FACTOR_WIDTH = 0.5
        self.SCALING_FACTOR_HEIGHT = 0.35
        self.SCALING_FACTOR_WIDTH = 0.5
        # dicts not because they have to be, but just so that
        # they are named because that might be useful
        # and because names are the only thing keeping horrendous this code together
        # TODO: current evil sharing of pointers (thanks python lmao) needs to be deleted
        # ofc delete from ik_env, retain here (they're in both spots rn)
        # NOTE: you need to do ax.draw(artist) for blitting. so each artist needs to be connected
        # to its particular artist. so they can't all be in one dict. 
        # we'll just increase the dict depth by 1, and add a named tuple to layer 1 (layer of axis)
        self.AxisAndArtists = namedtuple("AxAndArtists", "ax artists")
        self.axes_and_updating_artists = {}
        # TODO: maybe do the same for fixed artists. right now they're saved just to have them on hand
        # in case they'll be needed for something
        self.fixed_artists = {}
        # messes up rotating the 3d plot
        # you should catch the event and then draw
        # but i aint doing that
        self.blit = True

        ########################
        #  run related things  #
        ########################
        # TODO: remove from here,
        # we won't be generating runs here later
        if data == None: 
            self.n_iters = 200
        else:
            self.n_iters = data['qs'].shape[0]
        # the word time is used for timing
        self.t = np.arange(self.n_iters)
        # local kinematics integrator
        # but this thing handles plotting
        # only plotting needs to be run, so feel free to 
        # TODO butcher this later on.
        # NOTE: this thing holds axes and artists.
        # make this class hold them, because then it can manage
        # all of them in all figures, this is scitzo af bruv
        self.ik_env = InverseKinematicsEnv()
        # TODO: do this depending on the number of runs you'll be loading
        self.ik_env.damping = 25
        # putting it into this class so that python remembers it 'cos reasons, whatever
        # TODO: load this from run log files later
        # or just remove it since you're not generating runs from here
        self.controller1 = getController("")
        self.controller2 = getController("invKinm_Jac_T")
        # TODO: load runs, not make them
        # but deliver the same format.
        # TODO: ensure you're saving AT LEAST what's required here
        # NOTE: the jacobian svd is computed offline in these
        # TODO: make loading of multiple datas possible
        # TODO this data has 0 bussiness being defined in ik_env dude
        # it was probably there to get object permanence before this was a class,
        # but still dawg
        if data == None:
            self.ik_env.robots.append(Robot_raw(robot_name="no_sim"))
            self.ik_env.data.append(makeRun(self.controller1, self.ik_env, self.n_iters, 0))
            self.ik_env.data.append(makeRun(self.controller2, self.ik_env, self.n_iters, 1))
        else:
            self.ik_env.data.append(data)


        # ugly front end code is ugly.
        # i hate front end and this is why.
        # actual placing of plots in all this comes later because tkinter complains otherwise
        #####################################################
        #  LAYOUT OF THE GUI                                #
        # putting plots into: frames -> notebooks -> tabs   #
        #####################################################
        self.notebook_left = Notebook(root, height=int(self.SCREEN_HEIGHT))
        self.notebook_right = Notebook(root, height=int(self.SCREEN_HEIGHT))
        self.frame_manipulator = Frame(self.notebook_left)
        self.frame_manip_graphs = Frame(self.notebook_right)
        self.frame_imu = Frame(self.notebook_right)
        self.frame_ee = Frame(self.notebook_left)
        self.tabs_left = self.notebook_left.add(self.frame_manipulator, text='manipulator')
        self.tabs_left = self.notebook_left.add(self.frame_ee, text="end-effector")
        self.tabs_right = self.notebook_right.add(self.frame_manip_graphs, text='manipulator-graphs')
        self.tabs_right = self.notebook_right.add(self.frame_imu, text="ee-graphs")
        self.notebook_left.grid(row=0, column=0, sticky='ew')
        self.notebook_right.grid(row=0, column=1, sticky='ew')

        #######################################################################
        #                     PLACING ELEMENTS IN THE GUI                     #
        #######################################################################
        # NOTE: this thing is not used
        self.same_starting_position_on_off_var = IntVar()   
        self.same_starting_position_on_off_var.set(0)
        self.same_starting_position_checkbutton= Checkbutton(root, text = "same starting position on/off",
                              variable = self.same_starting_position_on_off_var,
                              onvalue = 1,
                              offvalue = 0,
        #                      command=same_starting_position_cmd,
                              )

        # LEFT PANE BELOW MANIP/EE PLOTS
        self.frame_below_manipulator_plot = Frame(self.frame_manipulator)
        self.frame_below_manipulator_plot.grid(column=0, row=3)

        # set controller 1
        self.frame_controller_menu1 = Frame(self.frame_below_manipulator_plot)
        self.controller_string1 = StringVar(self.frame_controller_menu1) 
        self.controller_string1.set("invKinm_dampedSquares") # default value 
        self.controller_menu1 = OptionMenu(self.frame_controller_menu1, self.controller_string1, 
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
        self.controller_string1.set("invKinm_dampedSquares")
        self.controller_menu1.grid(column=1, row=0)
        Label(self.frame_controller_menu1, text="Robot 1 controller:", background='yellow').grid(row=0, column=0, pady=4, padx = 4)
        self.frame_controller_menu1.grid(column=0, row=0)

        # set controller 2
        self.frame_controller_menu2 = Frame(self.frame_below_manipulator_plot)
        self.controller_string2 = StringVar(self.frame_controller_menu2) 
        self.controller_menu2 = OptionMenu(self.frame_controller_menu2, self.controller_string2, 
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
        self.controller_string2.set("invKinm_Jac_T") # default value 
        Label(self.frame_controller_menu1, text="Robot 1 controller:", background='black', foreground='white').grid(row=0, column=0, pady=4, padx = 4)
        self.controller_menu2.grid(column=1, row=0)
        Label(self.frame_controller_menu2, text="Robot 2 controller:", background='#EE82EE').grid(row=0, column=0, pady=4, padx = 4)
        self.frame_controller_menu2.grid(column=0, row=1)


        self.ellipse_on_off_var = IntVar()   
        self.ellipse_on_off_var.set(1)

        self.ellipse_checkbutton= Checkbutton(self.frame_below_manipulator_plot, text = "ellipse on/off",
                              variable = self.ellipse_on_off_var,
                              onvalue = 1,
                              offvalue = 0,
                              command=self.add_remove_ellipse,
                              )
        self.ellipse_checkbutton.grid(column=1, row=0)


        self.frame_goal_x = Frame(self.frame_below_manipulator_plot)
        Label(self.frame_goal_x, text="goal x").grid(row=0, column=0, pady=4, padx = 4)
        self.slider_goal_x = Scale(self.frame_goal_x, from_=-1.0, to=1.0, length=self.SLIDER_SIZE, orient=HORIZONTAL,
                                      command=self.update_goal_x)
        self.slider_goal_x.grid(column=1, row=0)
        self.frame_goal_x.grid(column=1, row=1)

        self.frame_goal_y = Frame(self.frame_below_manipulator_plot)
        Label(self.frame_goal_y, text="goal y").grid(row=0, column=0, pady=4, padx = 4)
        self.slider_goal_y = Scale(self.frame_goal_y, from_=-1.0, to=1.0, length=self.SLIDER_SIZE, orient=HORIZONTAL,
                                      command=self.update_goal_y)
        self.slider_goal_y.grid(column=1, row=0)
        self.frame_goal_y.grid(column=1, row=2)


        self.frame_goal_z = Frame(self.frame_below_manipulator_plot)
        Label(self.frame_goal_z, text="goal z").grid(row=0, column=0, pady=4, padx = 4)
        self.slider_goal_z = Scale(self.frame_goal_z, from_=-1.0, to=1.0, length=self.SLIDER_SIZE, orient=HORIZONTAL,
                                      command=self.update_goal_z)
        self.slider_goal_z.grid(column=1, row=0)
        self.frame_goal_z.grid(column=1, row=3)


        self.frame_update = Frame(self.frame_manip_graphs)
        Label(self.frame_update, text="Time").grid(row=0, column=0, pady=4, padx = 4)
        self.slider_update = Scale(self.frame_update, from_=0, to=self.n_iters - 1, length=self.SLIDER_SIZE, orient=HORIZONTAL,
                                      command=self.update_points)#, label="Frequency [Hz]")
        self.slider_update.grid(column=1, row=0)
        self.frame_update.grid(column=0, row=3)

        self.frame_update2 = Frame(self.frame_imu)
        Label(self.frame_update2, text="Time").grid(row=0, column=0, pady=4, padx = 4)
        self.slider_update = Scale(self.frame_update2, from_=0, to=self.n_iters - 1, length=self.SLIDER_SIZE, orient=HORIZONTAL,
                                      command=self.update_points)#, label="Frequency [Hz]")
        self.slider_update.grid(column=1, row=0)
        self.frame_update2.grid(column=0, row=3)


        self.button_quit = Button(master=self.frame_manip_graphs, text="Quit", command=root.destroy)
        self.button_reset = Button(master=self.frame_manip_graphs, text="New run", command=self.reset)
        self.button_play = Button(master=self.frame_manip_graphs, text="Play", command=self.play)
        self.button_play.grid(column=0, row=4)
        self.button_reset.grid(column=0, row=5)
        self.button_quit.grid(column=0, row=6)

        self.button_quit2 = Button(master=self.frame_imu, text="Quit", command=root.destroy)
        self.button_reset2 = Button(master=self.frame_imu, text="New run", command=self.reset)
        self.button_play2 = Button(master=self.frame_imu, text="Play", command=self.play)
        self.button_play2.grid(column=0, row=4)
        self.button_reset2.grid(column=0, row=5)
        self.button_quit2.grid(column=0, row=6)

        self.slider_goal_x.set(self.ik_env.goal[0])
        self.slider_goal_y.set(self.ik_env.goal[1])
        self.slider_goal_z.set(self.ik_env.goal[2])

        
        ################################
        #  ALL PLOTS ARE DEFINED HERE  #
        ################################

        # how to add plots will be documented below on an easier example
        #######################################################################
        #                             UNUSED PLOT                             #
        #######################################################################
        self.fig_ee = Figure(figsize=(self.SCREEN_WIDTH/self.DPI*self.SCALING_FACTOR_WIDTH, self.SCREEN_HEIGHT/self.DPI*self.SCALING_FACTOR_HEIGHT), dpi=self.DPI)
        self.rec3D_ax = self.fig_ee.add_subplot(projection='3d') 
        self.rec3D_ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
        # tkinterize
        self.canvas_ee = FigureCanvasTkAgg(self.fig_ee, master=self.frame_ee) 
        self.canvas_ee_widget = self.canvas_ee.get_tk_widget()     
        self.canvas_ee_widget.grid(row=0, column=0) 
        self.canvas_ee._tkcanvas.grid(row=1, column=0)   
        # draw before getting the background
        self.canvas_ee.draw()
        self.background_ee = self.canvas_ee.copy_from_bbox(self.fig_ee.bbox)
        # TODO: append artists to either fixed or updating artists dicts

        #######################################################################
        #                          MANIPULATOR PLOT                           #
        #######################################################################
        # robot plot
        self.fig_manipulator = Figure(figsize=(self.SCREEN_WIDTH/self.DPI*self.SCALING_FACTOR_WIDTH, self.SCREEN_HEIGHT/self.DPI*self.SCALING_FACTOR_HEIGHT), dpi=self.DPI)
        # TODO: fix evil pointer sharing between this class and ik_env
        self.ik_env.ax = self.fig_manipulator.add_subplot(projection='3d') 
        # read comment on top of init to see what this is
        self.axes_and_updating_artists["ax_manipulators"] = self.AxisAndArtists(self.ik_env.ax, {})
        # fix array sizes to be the workspace as that won't change. otherwise it's updated as you go,
        # and that's impossible to look at.
        self.ik_env.ax.plot(np.array([0]), np.array([0]), np.array([1.5]), c='b')
        self.ik_env.ax.plot(np.array([0]), np.array([0]), np.array([-1.5]), c='b')
        self.ik_env.ax.set_xlim([-1.0,1.0])
        self.ik_env.ax.set_ylim([-1.0,1.0])
        # TODO: generate this from a colormap
        self.link_colors = ['black', 'violet']
        self.trajectory_plots = []
        # this way you can have more robots plotted and see the differences between algorithms
        # but it of course works on a single robot too
        for robot_index, robot in enumerate(self.ik_env.robots):
            robot.initDrawing(self.ik_env.ax, self.link_colors[robot_index])
            # this could be named better, but i can't be bothered right now
            for i, link in enumerate(robot.lines):
                for j, line in enumerate(link):
                    # constuct a name that unique at least
                    self.axes_and_updating_artists["ax_manipulators"].artists["robot_" + str(robot_index) + "_link_" + str(i) + "_line_" + str(j)] = line
            # ee point
            ee_point_plot = drawPoint(self.ik_env.ax, self.ik_env.data[robot_index]['p_es'][0], 'red', 'o')
            self.ik_env.p_e_point_plots.append(ee_point_plot)
            self.axes_and_updating_artists["ax_manipulators"].artists["ee_point_plot"] = ee_point_plot
            # plot ee position trajectory. not blitted because it is fixed
            # TODO: blit it for real-time operation
            trajectory_plot, = self.ik_env.ax.plot(self.ik_env.data[robot_index]['p_es'][:,0], \
                    self.ik_env.data[robot_index]['p_es'][:,1], self.ik_env.data[robot_index]['p_es'][:,2], color='blue')
            # TODO stop having shared pointers all over the place, it's evil
            self.trajectory_plots.append(trajectory_plot)
            if self.real_time_flag:
                self.axes_and_updating_artists["ax_manipulators"].artists["trajectory_plot_" + str(robot_index)] = \
                        trajectory_plot
                raise NotImplementedError("real time ain't implemented yet, sorry")
            else:
                self.fixed_artists["trajectory_plot_" + str(robot_index)] = trajectory_plot

        # goal point
        # only makes sense for clik, but keep it here anyway, it doesn't hurt
        # TODO: add a button to turn it off
        goal_point_plot = drawPoint(self.ik_env.ax, self.ik_env.goal, 'maroon', '*')
        self.ik_env.goal_point_plot = goal_point_plot
        self.fixed_artists["goal_point_plot"] = goal_point_plot

        # the manipulability ellipses
        for robot_index, robot in enumerate(self.ik_env.robots):
            radii = 1.0/self.ik_env.data[robot_index]["manip_ell_eigenvals"][0] * 0.1
            u = np.linspace(0.0, 2.0 * np.pi, 60)     
            v = np.linspace(0.0, np.pi, 60)     
            x = radii[0] * np.outer(np.cos(u), np.sin(v))     
            y = radii[1] * np.outer(np.sin(u), np.sin(v))     
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for i in range(len(x)):         
                for j in range(len(x)):
                    # TODO: add if to compute this in real-time if you're running real-time
                    # although, again, primary purpose is to log stuff for later
                    [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], self.ik_env.data[robot_index]["manip_elip_svd_rots"][0]) + self.ik_env.data[robot_index]['p_es'][0]

            ellipsoid_surf_plot = self.ik_env.ax.plot_surface(x, y, z,  rstride=3, cstride=3,
                            color='pink', linewidth=0.1,
                            alpha=0.3, shade=True)
            self.ik_env.ellipsoid_surf_plots.append(ellipsoid_surf_plot)
            # TODO: current evil sharing of pointers (thanks python lmao) needs to be deleted
            # ofc delete from ik_env, retain here
# TODO: put back in if possible
#            self.axes_and_updating_artists["ax_manipulators"].artists["ellipsoid_surf_plot_"+ str(robot_index)] = \
#                    ellipsoid_surf_plot

        # tkinterize the figure/canvas
        self.canvas_manipulator = FigureCanvasTkAgg(self.fig_manipulator, master=self.frame_manipulator) 
        # need to hit it with the first draw before saving background
        self.canvas_manipulator.draw()
        # save background - in manual blitting you reload and update yourself
        self.background_manipulator = self.canvas_manipulator.copy_from_bbox(self.fig_manipulator.bbox)
        # take and put the matplotlib toolbar below the plot.
        # these are quite nifty (zooming, adjusting axes etc)
        self.canvas_manipulator_widget = self.canvas_manipulator.get_tk_widget()     
        self.canvas_manipulator_widget.grid(row=0, column=0) 
        self.canvas_manipulator._tkcanvas.grid(row=1, column=0)   
        # i probably don't need this draw, but who cares
        #self.canvas_manipulator.draw()


        #######################################################################
        #                        manipulability plots                         #
        #######################################################################
        # NOTE: code for each plot is replicated,
        # and there are more for loops than necessary.
        # but honestly this is more readable because you know which subplot you're making and with what.
        # TODO: maybe make it automatic later by making tabs etc automatic via dicts.
        # but as it stands right now, it's fine, i won't do it unless there is a distinct need for it.
        # manipulability ellipsoid eigenvalues
        self.fig_manip_graphs = Figure(figsize=(self.SCREEN_WIDTH/self.DPI*self.SCALING_FACTOR_WIDTH, self.SCREEN_HEIGHT/self.DPI*self.SCALING_FACTOR_HEIGHT), dpi=self.DPI)
        # NOTE: this ax object lives on in some other matplotlib object
        # so i don't have to save it
        ax_eigens = self.fig_manip_graphs.add_subplot(311)
        self.axes_and_updating_artists["ax_eigens"] = self.AxisAndArtists(ax_eigens, {})
        ax_eigens.set_title('Manipulability ellipsoid eigenvalues')
        ax_eigens.grid()
        ax_eigens.set_xticks([])
        self.eigen1_lines = []
        self.eigen2_lines = []
        self.eigen3_lines = []

        for robot_index, robot in enumerate(self.ik_env.robots):
            eigen1_line, = ax_eigens.plot(self.t, \
                    self.ik_env.data[robot_index]["manip_ell_eigenvals"][:,0], color=self.link_colors[robot_index])
            self.eigen1_lines.append(eigen1_line)
            eigen2_line, = ax_eigens.plot(self.t, \
                    self.ik_env.data[robot_index]["manip_ell_eigenvals"][:,1], color=self.link_colors[robot_index])
            self.eigen2_lines.append(eigen2_line)
            eigen3_line, = ax_eigens.plot(self.t, \
                    self.ik_env.data[robot_index]["manip_ell_eigenvals"][:,2], color=self.link_colors[robot_index])
            self.eigen3_lines.append(eigen3_line)
            if self.real_time_flag:
                # TODO: put it in updating artists, but then also actuall update it
                self.axes_and_updating_artists["ax_eigens"].artists["eigen1_line_" + str(robot_index)] = eigen1_line
                self.axes_and_updating_artists["ax_eigens"].artists["eigen2_line_" + str(robot_index)] = eigen2_line
                self.axes_and_updating_artists["ax_eigens"].artists["eigen3_line_" + str(robot_index)] = eigen3_line
                raise NotImplementedError("real time ain't implemented yet, sorry")
            else:
                self.fixed_artists["eigen1_line_" + str(robot_index)] = eigen1_line
                self.fixed_artists["eigen2_line_" + str(robot_index)] = eigen2_line
                self.fixed_artists["eigen3_line_" + str(robot_index)] = eigen3_line
        self.point_in_time_eigen1_line = ax_eigens.axvline(x=0, color='red', animated=True)
        self.axes_and_updating_artists["ax_eigens"].artists["point_in_time_eigen1_line"] = self.point_in_time_eigen1_line

        # manipulability index (volume of manipulability ellipsoid)
        ax_manips = self.fig_manip_graphs.add_subplot(312) 
        ax_manips.set_title('Manipulability index')
        self.axes_and_updating_artists["ax_manips"] = self.AxisAndArtists(ax_manips, {})
        ax_manips.grid()
        ax_manips.set_xticks([])
        self.manip_index_lines = []
        for robot_index, robot in enumerate(self.ik_env.robots):
            manip_index_line, = ax_manips.plot(self.t, self.ik_env.data[robot_index]['manip_indeces'], color=self.link_colors[robot_index])
            self.manip_index_lines.append(manip_index_line)
            if self.real_time_flag:
                # TODO: put it in updating artists, but then also actuall update it
                self.axes_and_updating_artists["ax_manips"].artists["manip_index_line_" + str(robot_index)] = manip_index_line
                raise NotImplementedError("real time ain't implemented yet, sorry")
            else:
                self.fixed_artists["manip_index_line_" + str(robot_index)] = manip_index_line
        self.point_in_time_manip_index_line = ax_manips.axvline(x=0, color='red', animated=True)
        self.axes_and_updating_artists["ax_manips"].artists["point_in_time_manip_index_line"] = self.point_in_time_manip_index_line

        # dist to goal (this could be/should be elsewhere)
        ax_goal_dists = self.fig_manip_graphs.add_subplot(313) 
        ax_goal_dists.set_title('Distance to goal')
        self.axes_and_updating_artists["ax_goal_dists"] = self.AxisAndArtists(ax_goal_dists, {})
        ax_goal_dists.grid()
        ax_goal_dists.set_xlabel('iter')
        self.dist_to_goal_lines = []
        for robot_index, robot in enumerate(self.ik_env.robots):
            dist_to_goal_line, = ax_goal_dists.plot(self.t, self.ik_env.data[robot_index]['dists_to_goal'], \
                    color=self.link_colors[robot_index])
            self.dist_to_goal_lines.append(dist_to_goal_line)
            if self.real_time_flag:
                # TODO: put it in updating artists, but then also actuall update it
                self.axes_and_updating_artists["ax_goal_dists"].artists["dist_to_goal_line" + str(robot_index)] = \
                        dist_to_goal_line
                raise NotImplementedError("real time ain't implemented yet, sorry")
            else:
                self.fixed_artists["dist_to_goal_line" + str(robot_index)] = dist_to_goal_line
        self.point_in_time_dist_to_goal_line = ax_goal_dists.axvline(x=0, color='red', animated=True)
        self.axes_and_updating_artists["ax_goal_dists"].artists["point_in_time_dist_to_goal_line"] = \
                self.point_in_time_dist_to_goal_line

        # tkinterize canvas for the whole figure (all subplots)
        self.canvas_manip_graphs = FigureCanvasTkAgg(self.fig_manip_graphs, master=self.frame_manip_graphs) 
        self.canvas_manip_graphs.draw()
        # save background for blitting
        self.background_manip_graphs = self.canvas_manip_graphs.copy_from_bbox(self.fig_manip_graphs.bbox)
        # put matplotlib toolbar below the plot
        self.canvas_manip_graphs_widget = self.canvas_manip_graphs.get_tk_widget()     
        self.canvas_manip_graphs_widget.grid(row=0, column=0) 
        self.canvas_manip_graphs._tkcanvas.grid(row=1, column=0)   
        #self.canvas_manip_graphs.draw()


        #######################################################################
        #                              IMU plots                              #
        #######################################################################
        self.fig_imu = Figure(figsize=(self.SCREEN_WIDTH/self.DPI*self.SCALING_FACTOR_WIDTH, self.SCREEN_HEIGHT/self.DPI*self.SCALING_FACTOR_HEIGHT), dpi=self.DPI)
        self.v_x_lines = []
        self.v_y_lines = []
        self.v_z_lines = []
        self.omega_x_lines = []
        self.omega_y_lines = []
        self.omega_z_lines = []

        ax_v_x = self.fig_imu.add_subplot(321)
        ax_v_x.grid()
        self.axes_and_updating_artists["ax_v_x"] = self.AxisAndArtists(ax_v_x, {})
        ax_v_y = self.fig_imu.add_subplot(322)
        ax_v_y.grid()
        self.axes_and_updating_artists["ax_v_y"] = self.AxisAndArtists(ax_v_y, {})
        ax_v_z = self.fig_imu.add_subplot(323)
        ax_v_z.grid()
        self.axes_and_updating_artists["ax_v_z"] = self.AxisAndArtists(ax_v_z, {})
        ax_omega_x = self.fig_imu.add_subplot(324)
        ax_omega_x.grid()
        self.axes_and_updating_artists["ax_omega_x"] = self.AxisAndArtists(ax_omega_x, {})
        ax_omega_y = self.fig_imu.add_subplot(325)
        ax_omega_y.grid()
        self.axes_and_updating_artists["ax_omega_y"] = self.AxisAndArtists(ax_omega_y, {})
        ax_omega_z = self.fig_imu.add_subplot(326)
        ax_omega_z.grid()
        self.axes_and_updating_artists["ax_omega_z"] = self.AxisAndArtists(ax_omega_z, {})
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
        for robot_index, robot in enumerate(self.ik_env.robots):
            v_x_line, = ax_v_x.plot(self.t, self.ik_env.data[robot_index]["vs"][:,0], color=self.link_colors[robot_index])
            v_y_line, = ax_v_y.plot(self.t, self.ik_env.data[robot_index]["vs"][:,1], color=self.link_colors[robot_index])
            v_z_line, = ax_v_z.plot(self.t, self.ik_env.data[robot_index]["vs"][:,2], color=self.link_colors[robot_index])
            omega_x_line, = ax_omega_x.plot(self.t, \
                    self.ik_env.data[robot_index]["vs"][:,3], color=self.link_colors[robot_index])
            omega_y_line, = ax_omega_y.plot(self.t, \
                    self.ik_env.data[robot_index]["vs"][:,4], color=self.link_colors[robot_index])
            omega_z_line, = ax_omega_z.plot(self.t, \
                    self.ik_env.data[robot_index]["vs"][:,5], color=self.link_colors[robot_index])
            self.v_x_lines.append(v_x_line)
            self.v_y_lines.append(v_y_line)
            self.v_z_lines.append(v_z_line)
            self.omega_x_lines.append(omega_x_line)
            self.omega_y_lines.append(omega_y_line)
            self.omega_z_lines.append(omega_z_line)
            if self.real_time_flag:
                # TODO: put it in updating artists, but then also actuall update it
                self.axes_and_updating_artists["ax_v_x"].artists["v_x_line" + str(robot_index)] = v_x_line
                self.axes_and_updating_artists["ax_v_z"].artists["v_y_line" + str(robot_index)] = v_y_line
                self.axes_and_updating_artists["ax_v_z"].artists["v_z_line" + str(robot_index)] = v_z_line
                self.axes_and_updating_artists["ax_omega_x"].artists["omega_x_line" + str(robot_index)] = omega_x_line
                self.axes_and_updating_artists["ax_omega_x"].artists["omega_y_line" + str(robot_index)] = omega_y_line
                self.axes_and_updating_artists["ax_omega_x"].artists["omega_z_line" + str(robot_index)] = omega_z_line
                raise NotImplementedError("real time ain't implemented yet, sorry")
            else:
                self.fixed_artists["v_x_line" + str(robot_index)] = v_x_line
                self.fixed_artists["v_y_line" + str(robot_index)] = v_y_line
                self.fixed_artists["v_z_line" + str(robot_index)] = v_z_line
                self.fixed_artists["omega_x_line" + str(robot_index)] = omega_x_line
                self.fixed_artists["omega_y_line" + str(robot_index)] = omega_y_line
                self.fixed_artists["omega_z_line" + str(robot_index)] = omega_z_line

        self.point_in_time_ax_v_x_line = ax_v_x.axvline(x=0, color='red', animated=True)
        self.point_in_time_ax_v_y_line = ax_v_y.axvline(x=0, color='red', animated=True)
        self.point_in_time_ax_v_z_line = ax_v_z.axvline(x=0, color='red', animated=True)
        self.point_in_time_ax_omega_x_line = ax_omega_x.axvline(x=0, color='red', animated=True)
        self.point_in_time_ax_omega_y_line = ax_omega_y.axvline(x=0, color='red', animated=True)
        self.point_in_time_ax_omega_z_line = ax_omega_z.axvline(x=0, color='red', animated=True)
        self.axes_and_updating_artists["ax_v_x"].artists["point_in_time_ax_v_x_line"] = self.point_in_time_ax_v_x_line
        self.axes_and_updating_artists["ax_v_z"].artists["point_in_time_ax_v_y_line"] = self.point_in_time_ax_v_y_line
        self.axes_and_updating_artists["ax_v_z"].artists["point_in_time_ax_v_z_line"] = self.point_in_time_ax_v_z_line
        self.axes_and_updating_artists["ax_omega_x"].artists["point_in_time_ax_omega_x_line"] = \
                self.point_in_time_ax_omega_x_line
        self.axes_and_updating_artists["ax_omega_x"].artists["point_in_time_ax_omega_y_line"] = \
                self.point_in_time_ax_omega_y_line
        self.axes_and_updating_artists["ax_omega_x"].artists["point_in_time_ax_omega_z_line"] = \
                self.point_in_time_ax_omega_z_line

        self.canvas_imu = FigureCanvasTkAgg(self.fig_imu, master=self.frame_imu) 
        self.canvas_imu.draw()
        self.background_imu = self.canvas_imu.copy_from_bbox(self.fig_imu.bbox)
        self.canvas_imu_widget = self.canvas_manip_graphs.get_tk_widget()     
        self.canvas_imu_widget.grid(row=0, column=0) 
        self.canvas_imu._tkcanvas.grid(row=1, column=0)   
        #self.canvas_imu.draw()

        # i don't even remember what these toolbars are lol
        # pack_toolbar=False will make it easier to use a layout manager later on.
        self.toolbar_manipulator = NavigationToolbar2Tk(self.canvas_manipulator, self.frame_manipulator, pack_toolbar=False)
        self.toolbar_manipulator.update()
        self.toolbar_manipulator.grid(column=0, row=2)
        self.toolbar_manip_graphs = NavigationToolbar2Tk(self.canvas_manip_graphs, self.frame_manip_graphs, pack_toolbar=False)
        self.toolbar_manip_graphs.update()
        self.toolbar_manip_graphs.grid(column=0, row=2)
        self.toolbar_ee = NavigationToolbar2Tk(self.canvas_ee, self.frame_ee, pack_toolbar=False)
        self.toolbar_ee.update()
        self.toolbar_ee.grid(column=0, row=2)
        self.toolbar_imu = NavigationToolbar2Tk(self.canvas_imu, self.frame_imu, pack_toolbar=False)
        self.toolbar_imu.update()
        self.toolbar_imu.grid(column=0, row=2)

        # update once to finish initialization
        self.update_points(0)

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
    def update_points(self, new_val):
        start = time.time()
        if self.blit:
            self.canvas_manipulator.restore_region(self.background_manipulator)
        # always blit the rest
        self.canvas_ee.restore_region(self.background_ee)
        self.canvas_manip_graphs.restore_region(self.background_manip_graphs)
        self.canvas_imu.restore_region(self.background_imu)
        index = int(np.floor(float(new_val)))
        # ee plot here, but NOTE: UNUSED

        # all these are in lists as that's what the line plot wants, 
        # despite the fact that we have single points
        self.point_in_time_eigen1_line.set_xdata([self.t[index]])
        self.point_in_time_manip_index_line.set_xdata([self.t[index]])
        self.point_in_time_dist_to_goal_line.set_xdata([self.t[index]])
        self.point_in_time_ax_v_x_line.set_xdata([self.t[index]])
        self.point_in_time_ax_v_y_line.set_xdata([self.t[index]])
        self.point_in_time_ax_v_z_line.set_xdata([self.t[index]])
        self.point_in_time_ax_omega_x_line.set_xdata([self.t[index]])
        self.point_in_time_ax_omega_y_line.set_xdata([self.t[index]])
        self.point_in_time_ax_omega_z_line.set_xdata([self.t[index]])

        self.ik_env.ax.set_title(str(index) + 'th iteration toward goal')
        # animate 3d manipulator
        for robot_index, robot in enumerate(self.ik_env.robots):
            # TODO: fix actually (put dh parameters of current robot in)
            #self.ik_env.robots[robot_index].setJoints(self.ik_env.data[robot_index]["qs"][index])
            self.ik_env.robots[robot_index].setJoints(self.ik_env.data[robot_index]["qs"][index][:6])
            self.ik_env.robots[robot_index].drawStateAnim()
#            for link in robot.lines:
#                for line in link:
#                    self.ik_env.ax.draw_artist(line)
            self.ik_env.p_e_point_plots[robot_index].set_data([self.ik_env.data[robot_index]['p_es'][index][0]], [self.ik_env.data[robot_index]['p_es'][index][1]])
            self.ik_env.p_e_point_plots[robot_index].set_3d_properties([self.ik_env.data[robot_index]['p_es'][index][2]])

            if self.ellipse_on_off_var.get():
                self.blit = False
                try:
                    self.ik_env.ellipsoid_surf_plots[robot_index].remove()
                except ValueError:
                    pass

                # ellipsoid update
                radii = 1.0/self.ik_env.data[robot_index]["manip_ell_eigenvals"][index] * 0.1
                u = np.linspace(0.0, 2.0 * np.pi, 60)     
                v = np.linspace(0.0, np.pi, 60)     
                x = radii[0] * np.outer(np.cos(u), np.sin(v))     
                y = radii[1] * np.outer(np.sin(u), np.sin(v))     
                z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
                for i in range(len(x)):         
                    for j in range(len(x)):
                        [x[i,j],y[i,j],z[i,j]] = \
                                np.dot([x[i,j],y[i,j],z[i,j]], self.ik_env.data[robot_index]["manip_elip_svd_rots"][index]) \
                                + self.ik_env.data[robot_index]['p_es'][index]
                self.ik_env.ellipsoid_surf_plots[robot_index] = self.ik_env.ax.plot_surface(x, y, z,  rstride=3, cstride=3,
                                color='pink', linewidth=0.1,
                                alpha=0.3, shade=True) 
            else:
                self.blit = True

        # now draw all updating artists all at once
        # for whatever reason it does not update robot lines, so i have that above
        for ax_key in self.axes_and_updating_artists:
            for artist_key in self.axes_and_updating_artists[ax_key].artists:
                self.axes_and_updating_artists[ax_key].ax.draw_artist(\
                        self.axes_and_updating_artists[ax_key].artists[artist_key])

        if self.blit:
            self.canvas_manipulator.blit(self.fig_manipulator.bbox)
            self.canvas_manipulator.flush_events()
        if not self.blit:
            self.canvas_manipulator.draw()
        # always blit the other ones, this is just because of the surf plot
        self.canvas_ee.blit(self.fig_ee.bbox)
        self.canvas_ee.flush_events()
        self.canvas_manip_graphs.blit(self.fig_manip_graphs.bbox)
        self.canvas_manip_graphs.flush_events()
        self.canvas_imu.blit(self.fig_imu.bbox)
        self.canvas_imu.flush_events()
#        self.canvas_imu.draw()
#        self.canvas_ee.draw()
        # TODO update may not be needed as we're going by slider here
        self.root.update()
        end = time.time()
        print("time per update:", end - start)
        print("fps", 1 / (end - start))


    def drawAll(self):
        self.canvas_ee.draw()
        self.canvas_manipulator.draw()
        self.canvas_manip_graphs.draw()
        self.canvas_imu.draw()

    def update_goal_x(self, new_val):
        goal_x = float(new_val)
        self.ik_env.goal[0] = goal_x
        self.ik_env.goal_point_plot.set_data([self.ik_env.goal[0]], [self.ik_env.goal[1]])
        self.ik_env.goal_point_plot.set_3d_properties([self.ik_env.goal[2]])
#        self.canvas_ee.draw()
        self.canvas_manipulator.draw()
#        self.canvas_manip_graphs.draw()
        # TODO update may not be needed as we're going by slider here
        self.root.update()

    def update_goal_y(self, new_val):
        goal_y = float(new_val)
        self.ik_env.goal[1] = goal_y
        self.ik_env.goal_point_plot.set_data([self.ik_env.goal[0]], [self.ik_env.goal[1]])
        self.ik_env.goal_point_plot.set_3d_properties([self.ik_env.goal[2]])
#        self.canvas_ee.draw()
        self.canvas_manipulator.draw()
#        self.canvas_manip_graphs.draw()
        # TODO update may not be needed as we're going by slider here
        self.root.update()

    def update_goal_z(self, new_val):
        goal_z = float(new_val)
        self.ik_env.goal[2] = goal_z
        self.ik_env.goal_point_plot.set_data([self.ik_env.goal[0]], [self.ik_env.goal[1]])
        self.ik_env.goal_point_plot.set_3d_properties([self.ik_env.goal[2]])
#        self.canvas_ee.draw()
        self.canvas_manipulator.draw()
#        self.canvas_manip_graphs.draw()
        # TODO update may not be needed as we're going by slider here
        self.root.update()


    def drawAndUpdateBackground(self):
        self.drawAll()
        self.background_ee = self.canvas_ee.copy_from_bbox(self.fig_ee.bbox)
        self.background_manipulator = self.canvas_manipulator.copy_from_bbox(self.fig_manipulator.bbox)
        self.background_manip_graphs = self.canvas_manip_graphs.copy_from_bbox(self.fig_manip_graphs.bbox)
        self.background_imu = self.canvas_imu.copy_from_bbox(self.fig_imu.bbox)

    def reset(self):
        self.ik_env.reset()
    #    ik_env.goal_point_plot.remove()
    #    ik_env.goal_point_plot = drawPoint(ik_env.ax, ik_env.goal, 'red', 'o')
        self.controller1 = getController(self.controller_string1.get())
        self.controller2 = getController(self.controller_string2.get())
        controllers = [self.controller1, self.controller2]
        for robot_index, robot in enumerate(self.ik_env.robots):
            self.ik_env.data.append(makeRun(controllers[robot_index], self.ik_env, self.n_iters, robot_index))
            self.trajectory_plots[robot_index].set_data(self.ik_env.data[robot_index]['p_es'][:,0], self.ik_env.data[robot_index]['p_es'][:,1])
            self.trajectory_plots[robot_index].set_3d_properties(self.ik_env.data[robot_index]['p_es'][:,2])
            self.eigen1_lines[robot_index].set_ydata(self.ik_env.data[robot_index]["manip_ell_eigenvals"][:,0])
            self.eigen2_lines[robot_index].set_ydata(self.ik_env.data[robot_index]["manip_ell_eigenvals"][:,1])
            self.eigen3_lines[robot_index].set_ydata(self.ik_env.data[robot_index]["manip_ell_eigenvals"][:,2])
            self.manip_index_lines[robot_index].set_ydata(self.ik_env.data[robot_index]['manip_indeces'])
            self.dist_to_goal_lines[robot_index].set_ydata(self.ik_env.data[robot_index]['dists_to_goal'])
        self.update_points(0)
        self.drawAndUpdateBackground()
        self.root.update()

# TODO: use the same API you use for real-time plotting:
# --> just manually put things into the queue here
    def play(self):
        for i in range(self.n_iters):
            self.update_points(i)

    # ellipse on/off
    def add_remove_ellipse(self):
        # this just deletes them le mao
        #        for artist_key in self.axes_and_updating_artists["ax_manipulators"].artists:
        #            if "robot" in artist_key:
        #                self.axes_and_updating_artists["ax_manipulators"].artists[artist_key].set_animated(bool(self.ellipse_on_off_var))

        try:
            for robot_index, robot in enumerate(self.ik_env.robots):
                self.ik_env.ellipsoid_surf_plots[robot_index].remove()
        except ValueError:
            pass
        self.drawAndUpdateBackground()
        self.root.update()



if __name__ == "__main__":
    queue = Queue()
    root = Tk()
    # TODO: change to something different obviously
    # or add a button to load or something, idc
    log_data_file_name = "/home/gospodar/lund/praxis/projects/ur_simple_control/python/examples/data/clik_run_001.pickle"
    args_file_name = "/home/gospodar/lund/praxis/projects/ur_simple_control/python/examples/data/clik_run_001_args.pickle"
    log_data, args = loadRunForAnalysis(log_data_file_name, args_file_name)
    args.visualize_manipulator = False
    log_data = cleanUpRun(log_data, log_data['qs'].shape[0], 200)
    robot = RobotManager(args)
    log_data = loadRun(args, robot, log_data)
    #gui = ManipulatorVisualMotionAnalyzer(root, queue, log_data, False)
    gui = ManipulatorVisualMotionAnalyzer(root, queue, None, False)

    # have mainloop 'cos from tkinter import *
    mainloop()
    # alternative if someone complains
    #root.mainloop()


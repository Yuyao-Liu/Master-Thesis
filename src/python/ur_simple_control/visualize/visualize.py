import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import time
import copy
#from pinocchio.visualize import MeshcatVisualizer
# TODO: use wrapped meshcat visualizer to have access to more nifty plotting
from ur_simple_control.visualize.meshcat_viewer_wrapper.visualizer import MeshcatVisualizer
from pinocchio.visualize import MeshcatVisualizer as UnWrappedMeshcat
import meshcat_shapes

# tkinter stuff for later reference
#from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from tkinter import *
#from tkinter.ttk import *

# rows and cols are flipped lel
def getNRowsMColumnsFromTotalNumber(n_plots):
    if n_plots == 1:
        n_cols = 1
        n_rows = 1
    if n_plots == 2:
        n_rows = 2
        n_cols = 1
    # i'm not going to bother with differently sized plots 
    if (n_plots == 3) or (n_plots == 4):
        n_cols = 2
        n_rows = 2
    if (n_plots == 5) or (n_plots == 6):
        n_cols = 2
        n_rows = 3
    if (n_plots >= 7) and (n_plots <= 9):
        n_cols = 3
        n_rows = 3
    if n_plots >= 10:
        raise NotImplementedError("sorry, you can only do up to 9 plots. more require tabs, and that's future work")
    return n_rows, n_cols


def plotFromDict(plot_data, final_iteration, args, title="title"):
    """ 
    plotFromDict
    ------------
    plots logs stored in a dictionary
    - every key is one subplot, and it's value
      is what you plot
    """ 
    # TODO: replace with actual time ( you know what dt is from args)
    t = np.arange(final_iteration)
    n_cols, n_rows = getNRowsMColumnsFromTotalNumber(len(plot_data))
    # this is what subplot wants
    subplot_col_row = str(n_cols) + str(n_rows)
    ax_dict ={}
    # NOTE: cutting off after final iterations is a vestige from a time
    # when logs were prealocated arrays, but it ain't hurtin' nobody as it is
    plt.title(title)
    for i, data_key in enumerate(plot_data):
        colors = plt.cm.jet(np.linspace(0, 1, plot_data[data_key].shape[1]))
        ax_dict[data_key] = plt.subplot(int(subplot_col_row + str(i + 1)))
        for j in range(plot_data[data_key].shape[1]):
            ax_dict[data_key].plot(t, plot_data[data_key][:final_iteration,j], color=colors[j], label=data_key + "_" + str(j))
        ax_dict[data_key].legend()
    plt.show()


# STUPID MATPLOTLIB CAN'T HANDLE MULTIPLE FIGURES FROM DIFFERENT PROCESS
# LITERALLY REFUSES TO GET ME A FIGURE
def realTimePlotter(args, log_item, queue):
    """
    realTimePlotter
    ---------------
    - true to its name
    - plots whatever you are logging if you use the --real-time-plotting flag
    """
    if args.debug_prints:
        print("REAL_TIME_PLOTTER: i got this queue:", queue)
    # NOTE: CAN'T PLOT ANYTHING BEFORE, THIS HANGS IF YOU HAD A PLOT BEFORE
    plt.ion()
    fig = plt.figure()
    canvas = fig.canvas
#    if args.debug_prints:
#        print("REAL_TIME_PLOTTER: putting success into queue")
    #queue.put("success")
    logs_deque = {}
    logs_ndarrays = {}
    AxisAndArtists = namedtuple("AxAndArtists", "ax artists")
    axes_and_updating_artists = {}
    #if args.debug_prints:
    #    print("REAL_TIME_PLOTTER: i am waiting for the first log_item to initialize myself")
    #log_item = queue.get()
    if len(log_item) == 0:
        print("you've send me nothing, so no real-time plotting for you")
        return
    #if args.debug_prints:
    #    print("REAL_TIME_PLOTTER: got log_item, i am initializing the desired plots")
    ROLLING_BUFFER_SIZE = 100
    t = np.arange(ROLLING_BUFFER_SIZE)

    n_cols, n_rows = getNRowsMColumnsFromTotalNumber(len(log_item))
    # this is what subplot wants
    subplot_col_row = str(n_cols) + str(n_rows)
    # preload some zeros and initialize plots
    for i, data_key in enumerate(log_item):
        # you give single-vector numpy arrays, i instantiate and plot lists of these 
        # so for your (6,) vector, i plot (N, 6) ndarrays resulting in 6 lines of length N.
        # i manage N because plot data =/= all data for efficiency reasons.
        assert type(log_item[data_key]) == np.ndarray
        assert len(log_item[data_key].shape) == 1
        # prepopulate with zeros via list comperhension (1 item is the array, the queue is 
        # ROLLING_BUFFER_SIZE of such arrays, and these go in and out at every time step)
        logs_deque[data_key] = deque([log_item[data_key] for index in range(ROLLING_BUFFER_SIZE)])
        # i can only plot np_arrays, so these queues will have to be turned to nparray at every timestep
        # thankfull, deque is an iterable
        logs_ndarrays[data_key] = np.array(logs_deque[data_key])
        colors = plt.cm.jet(np.linspace(0, 1, log_item[data_key].shape[0]))
        ax = fig.add_subplot(int(subplot_col_row + str(i + 1)))
        # some hacks, i'll have to standardize things somehow
        if 'qs' in data_key:
            ax.set_ylim(bottom=-6.14, top=6.14)
        if 'vs' in data_key:
            ax.set_ylim(bottom=-3.14, top=3.14)
        if 'wrench' in data_key:
            ax.set_ylim(bottom=-20.0, top=20.0)
        if 'tau' in data_key:
            ax.set_ylim(bottom=-2.0, top=2.0)
        if 'err' in data_key:
            ax.set_ylim(bottom=-2.0, top=2.0)
        axes_and_updating_artists[data_key] = AxisAndArtists(ax, {})
        for j in range(log_item[data_key].shape[0]):
            # the comma is because plot retuns ax, sth_unimportant.
            # and python let's us assign iterable return values like this
            axes_and_updating_artists[data_key].artists[str(data_key) + str(j)], = \
                    axes_and_updating_artists[data_key].ax.plot(t, logs_ndarrays[data_key][:,j], 
                                                             color=colors[j], label=data_key + "_" + str(j))
        axes_and_updating_artists[data_key].ax.legend(loc='upper left')

    # need to call it once
    canvas.draw()
    canvas.flush_events()
    background = fig.bbox

    if args.debug_prints:
        print("REAL_TIME_PLOTTER: FULLY ONLINE")
    try:
        while True:
            log_item = queue.get()
            if log_item == "befree":
                if args.debug_prints:
                    print("REAL_TIME_PLOTTER: got befree, realTimePlotter out")
                break
            for data_key in log_item:
                # remove oldest
                logs_deque[data_key].popleft()
                # put in new one
                logs_deque[data_key].append(log_item[data_key])
                # make it an ndarray (plottable)
                logs_ndarrays[data_key] = np.array(logs_deque[data_key])
                # now shape == (ROLLING_BUFFER_SIZE, vector_dimension)
                for j in range(logs_ndarrays[data_key].shape[1]):
                    axes_and_updating_artists[data_key].artists[str(data_key) + str(j)].set_data(t, logs_ndarrays[data_key][:,j])
                    axes_and_updating_artists[data_key].ax.draw_artist(\
                            axes_and_updating_artists[data_key].artists[str(data_key) + str(j)])
            canvas.blit(fig.bbox)
            canvas.flush_events()
    except KeyboardInterrupt:
        if args.debug_prints:
            print("REAL_TIME_PLOTTER: caught KeyboardInterrupt, i'm out")
    plt.close(fig)


def manipulatorVisualizer(model, collision_model, visual_model, args, cmd, queue):
    viz = MeshcatVisualizer(model=model, collision_model=collision_model, visual_model=visual_model)
    viz.loadViewerModel()
    # display the initial pose
    viz.display(cmd["q"])
    # set shapes we know we'll use
    meshcat_shapes.frame(viz.viewer["Mgoal"], opacity=0.5)
    meshcat_shapes.frame(viz.viewer["T_w_e"], opacity=0.5)
    meshcat_shapes.frame(viz.viewer["T_base"], opacity=0.5)
    print("MANIPULATORVISUALIZER: FULLY ONLINE")
    try:
        while True:
            cmd = queue.get()
            for key in cmd:
                if key == "befree":
                    if args.debug_prints:
                        print("MANIPULATORVISUALIZER: got befree, manipulatorVisualizer out")
                    viz.viewer.window.server_proc.kill()
                    viz.viewer.window.server_proc.wait()
                    break
                if key == "Mgoal":
                    viz.viewer["Mgoal"].set_transform(cmd["Mgoal"].homogeneous)
                if key == "T_w_e":
                    viz.viewer["T_w_e"].set_transform(cmd["T_w_e"].homogeneous)
                if key == "T_base":
                    viz.viewer["T_base"].set_transform(cmd["T_base"].homogeneous)
                if key == "q":
                    viz.display(cmd["q"])
                if key == "point":
                    viz.addPoint(cmd["point"])
                if key == "obstacle_sphere":
                    # stupid and evil but there is no time
                    viz.addSphereObstacle(cmd["obstacle_sphere"][0], cmd["obstacle_sphere"][1])
                if key == "obstacle_box":
                    # stupid and evil but there is no time
                    viz.addBoxObstacle(cmd["obstacle_box"][0], cmd["obstacle_box"][1])
                if key == "path":
                    # stupid and evil but there is no time
                    viz.addPath("", cmd["path"])
                if key == "frame_path":
                    # stupid and evil but there is no time
                    viz.addFramePath("", cmd["frame_path"])

    except KeyboardInterrupt:
        if args.debug_prints:
            print("MANIPULATORVISUALIZER: caught KeyboardInterrupt, i'm out")
        viz.viewer.window.server_proc.kill()
        viz.viewer.window.server_proc.wait()

# could be merged with the above function.
# but they're different enough in usage to warrent a different function,
# instead of polluting the above one with ifs
def manipulatorComparisonVisualizer(model, collision_model, visual_model, args, cmd, cmd_queue, ack_queue):
    for geom in visual_model.geometryObjects:
        if "hand" in geom.name:
            geom.meshColor = np.array([0.2,0.2,0.2,0.2])
    viz = MeshcatVisualizer(model=model, 
                            collision_model=collision_model, 
                            visual_model=visual_model)
    viz.initViewer(open=True)
    # load the first one
    viz.loadViewerModel(collision_color=[0.2,0.2,0.2,0.6])
    #viz.viewer["pinocchio/visuals"].set_property("visible", False)
    #viz.viewer["pinocchio/collisions"].set_property("visible", True)
    viz.displayVisuals(False)
    viz.displayCollisions(True)
    # maybe needed
    #viz.displayVisuals(True)
#    meshpath = viz.viewerVisualGroupName 
#    for geom in visual_model(6):
#        meshpath_i = meshpath + "/" + geometry_object.name
    #viz.viewer["pinocchio/visuals"].set_property("opacity", 0.2)
    #viz.viewer["pinocchio"].set_property("opacity", 0.2)
    #viz.viewer["pinocchio/visuals/forearm_link_0"].set_property("opacity", 0.2)

    #viz.viewer["pinocchio"].set_property("color", (0.2,0.2,0.2,0.2))
    #viz.viewer["pinocchio/visuals"].set_property("color",(0.2,0.2,0.2,0.2))
    #viz.viewer["pinocchio/visuals/forearm_link_0"].set_property("color", (0.2,0.2,0.2,0.2))

    ## this is the path we want, with the /<object> at the end
    #node = viz.viewer["pinocchio/visuals/forearm_link_0/<object>"]
    #print(node)
    #node.set_property("opacity", 0.2)
    #node.set_property("modulated_opacity", 0.2)
    #node.set_property("color", [0.2] * 4)
    #node.set_property("scale", 100.2)
    # this one actually works
    #node.set_property("visible", False)

    node = viz.viewer["pinocchio/visuals"]
    #node.set_property("visible", False)
    node.set_property("modulated_opacity", 0.4)
    node.set_property("opacity", 0.2)
    node.set_property("color", [0.2] * 4)
    
    #meshcat->SetProperty("path/to/my/thing/<object>", "opacity", alpha);


    # other robot display
    viz2 = UnWrappedMeshcat(model=model, 
                             collision_model=collision_model, 
                             visual_model=visual_model)
    viz2.initViewer(viz.viewer)
    # i don't know if rootNodeName does anything apart from being different
    #viz2.loadViewerModel(rootNodeName="pinocchio2", visual_color=(1.0,1.0,1.0,0.1))
    viz2.loadViewerModel(rootNodeName="pinocchio2")
    # initialize
    q1, q2 = cmd_queue.get()
    viz.display(q1)
    viz2.display(q2)

    ack_queue.put("ready")
    print("MANIPULATOR_COMPARISON_VISUALIZER: FULLY ONLINE")
    try:
        while True:
            q = cmd_queue.get()
            if type(q) == str:
                print("got str q")
                if q == "befree":
                    if args.debug_prints:
                        print("MANIPULATOR_COMPARISON_VISUALIZER: got befree, manipulatorComparisonVisualizer out")
                    viz.viewer.window.server_proc.kill()
                    viz.viewer.window.server_proc.wait()
                    break
            q1, q2 = q
            viz.display(q1)
            viz2.display(q2)
            # this doesn't really work because meshcat is it's own server
            # and display commands just relay their command and return immediatelly.
            # but it's better than nothing. 
            # NOTE: if there's lag in meshcat, just add a small sleep here before the 
            # ack signal - that will ensure synchronization because meshat will actually be ready
            ack_queue.put("ready")
    except KeyboardInterrupt:
        if args.debug_prints:
            print("MANIPULATOR_COMPARISON_VISUALIZER: caught KeyboardInterrupt, i'm out")
        viz.viewer.window.server_proc.kill()
        viz.viewer.window.server_proc.wait()


# TODO: this has to be a class so that we can access
# the canvas in the onclik event function
# because that can be a method of that class and 
# get arguments that way.
# but otherwise i can't pass arguments to that.
# even if i could with partial, i can't return them,
# so no cigar from that.
# NOTE this is being tried out in realtimelogplotter branch
# there is no successful re-drawing here.
def logPlotter(log, args, cmd, cmd_queue, ack_queue):
    """
    logPlotter
    ---------------
    - plots whatever you want as long as you pass the data
      as a dictionary where the key will be the name on the plot, 
      and the value have to be the dependent variables - 
      the independent variable is time and has to be the same for all items.
      if you want to plot something else, you need to write your own function.
      use this as a skeleton if you want that new plot to be updating too
      as then you don't need to think about IPC - just use what's here.
    - this might be shoved into a tkinter gui if i decide i really need buttons
    """
    if len(log) == 0:
        print("you've send me nothing, so no real-time plotting for you")
        return

    plt.ion()
    fig = plt.figure()
    canvas = fig.canvas
    AxisAndArtists = namedtuple("AxAndArtists", "ax artists")
    axes_and_updating_artists = {}

    n_cols, n_rows = getNRowsMColumnsFromTotalNumber(len(log))
    # this is what subplot wants
    subplot_col_row = str(n_cols) + str(n_rows)
    # preload some zeros and initialize plots
    for i, data_key in enumerate(log):
        # you give single-vector numpy arrays, i instantiate and plot lists of these 
        # so for your (6,) vector, i plot (N, 6) ndarrays resulting in 6 lines of length N.
        # i manage N because plot data =/= all data for efficiency reasons.
        assert type(log[data_key]) == np.ndarray

        colors = plt.cm.jet(np.linspace(0, 1, log[data_key].shape[1]))
        ax = fig.add_subplot(int(subplot_col_row + str(i + 1)))
        ax.set_title(data_key)
        # we plot each line separately so that they have different colors
        # we assume (N_timesteps, your_vector) shapes.
        # values do not update
        for j in range(log[data_key].shape[1]):
            # NOTE the same length assumption plays a part for correctness,
            # but i don't want that to be an error in case you know what you're doing
            ax.plot(np.arange(len(log[data_key])), log[data_key][:,j], 
                        color=colors[j], label=data_key + "_" + str(j))

        # vertical bar does update
        point_in_time_line = ax.axvline(x=0, color='red', animated=True)
        axes_and_updating_artists[data_key] = AxisAndArtists(ax, point_in_time_line)
        axes_and_updating_artists[data_key].ax.legend(loc='upper left')

    # need to call it once to start, more if something breaks
    canvas.draw()
    canvas.flush_events()
    background = canvas.copy_from_bbox(fig.bbox)

    # we need to have an event that triggers redrawing
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    def onEvent(event):
        print("drawing")
        canvas.draw()
        canvas.flush_events()
        background = canvas.copy_from_bbox(fig.bbox)
        print("copied canvas")

    cid = fig.canvas.mpl_connect('button_press_event', onEvent)


    ack_queue.put("ready")
    if args.debug_prints:
        print("LOG_PLOTTER: FULLY ONLINE")
    try:
        counter = 0
        while True:
            counter += 1
            time_index = cmd_queue.get()
            if time_index == "befree":
                if args.debug_prints:
                    print("LOG_PLOTTER: got befree, logPlotter out")
                break
            canvas.restore_region(background)
            for data_key in log:
                axes_and_updating_artists[data_key].artists.set_xdata([time_index])
                axes_and_updating_artists[data_key].ax.draw_artist(axes_and_updating_artists[data_key].artists)
            # NOTE: this is stupid, i just want to catch the resize event
            #if not (counter % 50 == 0):
            if True:
                canvas.blit(fig.bbox)
                canvas.flush_events()
            else:
                print("drawing")
                canvas.draw()
                canvas.flush_events()
                background = canvas.copy_from_bbox(fig.bbox)
                print("copied canvas")
            ack_queue.put("ready")
    except KeyboardInterrupt:
        if args.debug_prints:
            print("LOG_PLOTTER: caught KeyboardInterrupt, i'm out")
    plt.close(fig)

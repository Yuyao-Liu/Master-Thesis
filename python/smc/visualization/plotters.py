import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple


# rows and cols are flipped lel
def getNRowsMColumnsFromTotalNumber(n_plots):
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
        raise NotImplementedError(
            "sorry, you can only do up to 9 plots. more require tabs, and that's future work"
        )
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
    ax_dict = {}
    # NOTE: cutting off after final iterations is a vestige from a time
    # when logs were prealocated arrays, but it ain't hurtin' nobody as it is
    plt.title(title)
    for i, data_key in enumerate(plot_data):
        colors = plt.cm.jet(np.linspace(0, 1, plot_data[data_key].shape[1]))
        ax_dict[data_key] = plt.subplot(int(subplot_col_row + str(i + 1)))
        for j in range(plot_data[data_key].shape[1]):
            ax_dict[data_key].plot(
                t,
                plot_data[data_key][:final_iteration, j],
                color=colors[j],
                label=data_key + "_" + str(j),
            )
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
    # queue.put("success")
    logs_deque = {}
    logs_ndarrays = {}
    AxisAndArtists = namedtuple("AxAndArtists", "ax artists")
    axes_and_updating_artists = {}
    # if args.debug_prints:
    #    print("REAL_TIME_PLOTTER: i am waiting for the first log_item to initialize myself")
    # log_item = queue.get()
    if len(log_item) == 0:
        print("you've send me nothing, so no real-time plotting for you")
        return
    # if args.debug_prints:
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
        logs_deque[data_key] = deque(
            [log_item[data_key] for _ in range(ROLLING_BUFFER_SIZE)]
        )
        # i can only plot np_arrays, so these queues will have to be turned to nparray at every timestep
        # thankfull, deque is an iterable
        logs_ndarrays[data_key] = np.array(logs_deque[data_key])
        colors = plt.cm.jet(np.linspace(0, 1, log_item[data_key].shape[0]))
        ax = fig.add_subplot(int(subplot_col_row + str(i + 1)))
        # some hacks, i'll have to standardize things somehow
        if "qs" in data_key:
            ax.set_ylim(bottom=-6.14, top=6.14)
        if "vs" in data_key:
            ax.set_ylim(bottom=-3.14, top=3.14)
        if "wrench" in data_key:
            ax.set_ylim(bottom=-20.0, top=20.0)
        if "tau" in data_key:
            ax.set_ylim(bottom=-2.0, top=2.0)
        if "err" in data_key:
            ax.set_ylim(bottom=-2.0, top=2.0)
        axes_and_updating_artists[data_key] = AxisAndArtists(ax, {})
        for j in range(log_item[data_key].shape[0]):
            # the comma is because plot retuns ax, sth_unimportant.
            # and python let's us assign iterable return values like this
            (
                axes_and_updating_artists[data_key].artists[str(data_key) + str(j)],
            ) = axes_and_updating_artists[data_key].ax.plot(
                t,
                logs_ndarrays[data_key][:, j],
                color=colors[j],
                label=data_key + "_" + str(j),
            )
        axes_and_updating_artists[data_key].ax.legend(loc="upper left")

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
                # TODO: have some asserts here for more informative errors if something
                # incorrect was passed as the log item.
                # ideally this is done only once, not every time
                logs_ndarrays[data_key] = np.array(logs_deque[data_key])
                # now shape == (ROLLING_BUFFER_SIZE, vector_dimension)
                for j in range(logs_ndarrays[data_key].shape[1]):
                    axes_and_updating_artists[data_key].artists[
                        str(data_key) + str(j)
                    ].set_data(t, logs_ndarrays[data_key][:, j])
                    axes_and_updating_artists[data_key].ax.draw_artist(
                        axes_and_updating_artists[data_key].artists[
                            str(data_key) + str(j)
                        ]
                    )
            canvas.blit(fig.bbox)
            canvas.flush_events()
    except KeyboardInterrupt:
        if args.debug_prints:
            print("REAL_TIME_PLOTTER: caught KeyboardInterrupt, i'm out")
    plt.close(fig)


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
            ax.plot(
                np.arange(len(log[data_key])),
                log[data_key][:, j],
                color=colors[j],
                label=data_key + "_" + str(j),
            )

        # vertical bar does update
        point_in_time_line = ax.axvline(x=0, color="red", animated=True)
        axes_and_updating_artists[data_key] = AxisAndArtists(ax, point_in_time_line)
        axes_and_updating_artists[data_key].ax.legend(loc="upper left")

    # need to call it once to start, more if something breaks
    canvas.draw()
    canvas.flush_events()
    background = canvas.copy_from_bbox(fig.bbox)

    # we need to have an event that triggers redrawing
    # cid = fig.canvas.mpl_connect('button_press_event', onclick)
    def onEvent(event):
        print("drawing")
        canvas.draw()
        canvas.flush_events()
        background = canvas.copy_from_bbox(fig.bbox)
        print("copied canvas")

    cid = fig.canvas.mpl_connect("button_press_event", onEvent)

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
                axes_and_updating_artists[data_key].ax.draw_artist(
                    axes_and_updating_artists[data_key].artists
                )
            # NOTE: this is stupid, i just want to catch the resize event
            # if not (counter % 50 == 0):
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

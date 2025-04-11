import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as plt_col


def pathVisualizer(x0, goal, map_as_list, positions, path_gen):
    # plotting
    fig = plt.figure()
    handle_goal = plt.plot(*pg, c="g")[0]
    handle_init = plt.plot(*x0[:2], c="b")[0]
    handle_curr = plt.plot(
        *x0[:2], c="r", marker=(3, 0, np.rad2deg(x0[2] - np.pi / 2)), markersize=10
    )[0]
    handle_curr_dir = plt.plot(
        0, 0, marker=(2, 0, np.rad2deg(0)), markersize=5, color="w"
    )[0]
    handle_path = plt.plot([], [], c="k")[0]
    coll = []
    for map_element in map_as_list:
        coll.append(plt_col.PolyCollection(np.array(map_element)))
    plt.gca().add_collection(coll)
    handle_title = plt.text(
        5, 9.5, "", bbox={"facecolor": "w", "alpha": 0.5, "pad": 5}, ha="center"
    )
    plt.gca().set_aspect("equal")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.draw()

    # do the updating plotting
    for x in positions:
        handle_curr.set_data([x[0]], [x[1]])
        handle_curr.set_marker((3, 0, np.rad2deg(x[2] - np.pi / 2)))
        handle_curr_dir.set_data([x[0]], [x[1]])
        handle_curr_dir.set_marker((2, 0, np.rad2deg(x[2] - np.pi / 2)))
        handle_path.set_data([path_gen.target_path[::2], path_gen.target_path[1::2]])
        handle_title.set_text(f"{t:5.3f}")
        fig.canvas.draw()
        plt.pause(0.005)
    plt.show()

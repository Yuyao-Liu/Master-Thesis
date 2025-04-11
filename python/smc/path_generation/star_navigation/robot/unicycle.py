import numpy as np
from ..robot import MobileRobot
import matplotlib.pyplot as plt


def get_triangle_vertices(center, orientation, side_length):
    # calculate the distance from center to vertex
    dist = side_length / (2 * np.sin(np.radians(60)))

    # calculate the x and y offsets from center to vertex
    x_offset = dist * np.cos(orientation)
    y_offset = dist * np.sin(orientation)

    # calculate the three vertices
    vertex1 = (center[0] + x_offset, center[1] + y_offset)
    vertex2 = (
        center[0]
        + x_offset * np.cos(np.radians(120))
        - y_offset * np.sin(np.radians(120)),
        center[1]
        + x_offset * np.sin(np.radians(120))
        + y_offset * np.cos(np.radians(120)),
    )
    vertex3 = (
        center[0]
        + x_offset * np.cos(np.radians(240))
        - y_offset * np.sin(np.radians(240)),
        center[1]
        + x_offset * np.sin(np.radians(240))
        + y_offset * np.cos(np.radians(240)),
    )

    return (vertex1, vertex2, vertex3)


class Unicycle(MobileRobot):

    def __init__(self, width, vel_min=None, vel_max=None, name="robot"):
        self.vmax = vel_max[0]
        super().__init__(
            nu=2, nx=3, width=width, name=name, u_min=vel_min, u_max=vel_max
        )

    def f(self, x, u):
        return [u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]]  # vx  # vy  # wz

    def move(self, x, u, dt):
        u_sat = np.clip(u, self.u_min, self.u_max)
        x_next = x + np.array(self.f(x, u_sat)) * dt
        x_next = np.clip(x_next, self.x_min, self.x_max)
        while x_next[2] > 2 * np.pi:
            x_next[2] -= 2 * np.pi
        while x_next[2] <= -0 * np.pi:
            x_next[2] += 2 * np.pi
        return x_next, u_sat

    def h(self, x):
        return x[:2]  # x, y

    def vel_min(self):
        return self.u_min

    def vel_max(self):
        return self.u_max

    def init_plot(self, ax=None, color="b", alpha=0.7, markersize=10):
        handles, ax = super(Unicycle, self).init_plot(ax=ax, color=color, alpha=alpha)
        handles += ax.plot(
            0, 0, marker=(3, 0, np.rad2deg(0)), markersize=markersize, color=color
        )
        handles += ax.plot(
            0, 0, marker=(2, 0, np.rad2deg(0)), markersize=0.5 * markersize, color="w"
        )
        return handles, ax

    def update_plot(self, x, handles):
        super(Unicycle, self).update_plot(x, handles)
        handles[1].set_data([x[0]], [x[1]])
        handles[1].set_marker((3, 0, np.rad2deg(x[2] - np.pi / 2)))
        handles[1].set_markersize(handles[1].get_markersize())
        handles[2].set_data([x[0]], [x[1]])
        handles[2].set_marker((2, 0, np.rad2deg(x[2] - np.pi / 2)))
        handles[2].set_markersize(handles[2].get_markersize())

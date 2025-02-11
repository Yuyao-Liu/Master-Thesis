import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colr


def drawOrientation(ax, orientation, t_v, avg_link_lenth):
    # every line is drawn be specifing its x, y and z coordinates
    # thus we will take each relevant vector and turn every one of its coordinates into a steps array
    # and every vector will be translated accordingly
    # first we need to take rot mat and p out of hom mat
    steps = np.arange(0.0, 1.0, 0.1) * (avg_link_lenth / 2)
    
    # now we draw the orientation of the current frame
    col = ['b', 'g', 'r']
    for i in range(0,3):
        x = t_v[0] + orientation[i,0] * steps
        y = t_v[1] + orientation[i,1] * steps
        z = t_v[2] + orientation[i,2] * steps
        ax.plot(x, y, z, color=col[i])


def drawVector(ax, link, t_v, color_link):
    # first let's draw the translation vector to the next frame
    steps = np.arange(0.0, 1.0, 0.1)
    x = t_v[0] + link[0] * steps
    y = t_v[1] + link[1] * steps
    z = t_v[2] + link[2] * steps
    ax.plot(x, y, z, color=color_link)



def drawPoint(ax, p, color_inside, marker):    
    point, = ax.plot([p[0]], [p[1]], [p[2]], markerfacecolor=color_inside, markeredgecolor=color_inside, marker=marker, markersize=5.5, alpha=0.9)
    return point

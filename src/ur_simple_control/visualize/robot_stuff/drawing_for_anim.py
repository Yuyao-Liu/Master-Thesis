import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colr


def drawOrientationAnim(ax, orientation, orientation_lines, t_v, avg_link_lenth):
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
#        ax.plot(x, y, z, color=col[i])

        orientation_lines[i].set_data(x, y)
        orientation_lines[i].set_3d_properties(z)

def drawVectorAnim(ax, link, link_line, t_v, color_link):
    # first let's draw the translation vector to the next frame
    steps = np.arange(0.0, 1.0, 0.1)
    x = t_v[0] + link[0] * steps
    y = t_v[1] + link[1] * steps
    z = t_v[2] + link[2] * steps
    #ax.plot(x, y, z, color=color_link)
    link_line.set_data(x, y)
    link_line.set_3d_properties(z)





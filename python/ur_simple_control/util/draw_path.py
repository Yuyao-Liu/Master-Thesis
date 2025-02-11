"""
possible improvement:
- draw multiple lines
- then you would just generate multiple dmps for each trajectory
  and do movel's + movej's to provide the connections
possible improvement: make it all bezier curves
  https://matplotlib.org/stable/users/explain/artists/paths.html
  look at the example for path handling if that's what you'll need
    - not really needed, especially because we want hard edges to test our controllers
      (but if that was a parameter that would be ok i guess)
"""

import numpy as np
import matplotlib.pyplot as plt
# LassoSelector is used for drawing.
# The class actually just does the drawing with mouse input,
# and the rest of processing is done on the obtained path.
# Thus it is the correct tool for the job and there's no need
# to reimplement it from mouse events.
from matplotlib.widgets import LassoSelector
# Path is the most generic matplotlib class.
# It has some convenient functions to handle the draw line,
# i.e. collection of points, but just refer to matplotlib 
# documentation and examples on Lasso and Path to see more.
#from matplotlib.path import Path

class DrawPathManager:
    def __init__(self, args, ax):
        self.canvas = ax.figure.canvas
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.args = args

    def onselect(self, verts):
        # verts is a list of tuples
        self.path = np.array( [ [i[0], i[1]] for i in verts ] )
        # this would refresh the canvas and remove the drawing
        #self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


    # function we bind to key press even
    # made to save and exit
    def accept(self, event):
        if event.key == "enter":
            if self.args.debug_prints:
                print("pixel path:")
                print(self.path)
            self.disconnect()
            np.savetxt("./path_in_pixels.csv", self.path, delimiter=',', fmt='%.5f')
            # plt.close over exit so that we can call this from elsewhere and not kill the program
            plt.close()

def drawPath(args):
    # normalize both x and y to 0-1 range
    # we can multiply however we want later
    # idk about the number of points, but it's large enough to draw
    # a smooth curve on the screen, so it's enough for the robot to draw as well i assume
    # --> it is, and the system operates well
    # depending on the curve the number of points is in the 50-200 range
    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    selector = DrawPathManager(args, ax)

    # map key press to our function
    # thankfully it didn't mind have self as the first argument
    fig.canvas.mpl_connect("key_press_event", selector.accept)
    ax.set_title("The drawing has to be 1 continuous line. Press 'Enter' to accept the drawn path. ")
    plt.show()
    return selector.path


if __name__ == '__main__':
    args = get_args()
    drawPath(args)
    #plt.ion()
    #fig = plt.figure()
    #canvas = fig.canvas
    #ax = fig.add_subplot(111)
    #ax.plot(np.arange(100), np.sin(np.arange(100)))
    #canvas.draw()
    #canvas.flush_events()

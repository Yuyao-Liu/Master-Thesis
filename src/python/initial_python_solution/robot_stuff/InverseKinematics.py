# kinematics related imports
from robot_stuff.forw_kinm import *
from robot_stuff.inv_kinm import *
from robot_stuff.follow_curve import *
from robot_stuff.utils import *

# general imports
import numpy as np
import matplotlib.pyplot as plt

# ######################################### NOTE #########################
# i'm half-adding another robot for the purposes of the plot.
# this means that half of the code will end up being broken (as a lot of it is already anyway).
# in case you want to use this elsewhere, just find the commit before the change
# and run with that.
##################################################################


class InverseKinematicsEnv:

# set damping (i.e. dt i.e. precision let's be real)
    def __init__(self, model_path=None, initial_qpos=None, n_actions=None, n_substeps=None ):
        print('env created')
        self.chill_reset = False
        # TODO write a convenience dh_parameter loading function
        self.robot = Robot_raw(robot_name="no_sim")
        self.robots = [self.robot]
        self.init_qs_robot = []
        self.damping = 5
        self.error_vec = None
        self.n_of_points_done = 0
        # keep track of the timesteps (to be reset after every episode)
        self.n_of_tries_for_point = 0
        # for storing whole runs of all robots
        self.data = []
        self.p_e_point_plots = []
        self.ellipsoid_surf_plots = []
        self.same_starting_position_on_off = 0

        # init goal
        self.goal = np.random.random(3) * 0.7
        # needed for easy initialization of the observation space

        # TODO enable setting the other one with greater ease
        self.reward_type = 'dense'
        #self.reward_type = 'sparse'
        self.episode_score = 0


# NOTE: BROKEN AS IT'S FOR ONLY ONE ROBOT
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if self.reward_type == 'sparse':
            if not error_test(self.robot.p_e, self.goal):
                return np.float32(-1.0)
            else:
                return np.float32(10.0)
        if self.reward_type == 'dense':
            distance = goal_distance(achieved_goal, goal)
            if not error_test(self.robot.p_e, self.goal):
                #reward = -1 * distance + 1 / distance
                reward = -1 * distance 
            else:
                reward = 100
            return reward
            


    def simpleStep(self, q_dots, damping, index):
        self.robot.forwardKinmViaPositions(q_dots / self.damping, damping)
        self.n_of_tries_for_point += 1
#        done = False
#        if error_test(self.robot.p_e, self.goal):
#            info = {
#                'is_success': np.float32(1.0),
#            }
#        else:
#            info = {
#                'is_success': np.float32(0.0),
#            }
#        return done


# NOTE: BROKEN AS IT'S FOR ONLY ONE ROBOT
    def step(self, action):
        self.robot.forwardKinmViaPositions(action[:-1] / self.damping, action[-1])
        self.n_of_tries_for_point += 1
        obs = self._get_obs()

        done = False
        if error_test(self.robot.p_e, self.goal):
            info = {
                'is_success': np.float32(1.0),
            }
        else:
            info = {
                'is_success': np.float32(0.0),
            }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        self.episode_score += reward
        return obs, reward, done, info




    def reset(self):
        # TODO: initialize robot joints state to a random (but valid (in joint range)) initial state
        #       when using joint clamping
        self.episode_score = 0
        self.n_of_points_done += 1
        self.n_of_tries_for_point = 0
        self.data = []

        # generate new point
        # NOTE: simply taken away 'cos now it's set with a slider
        #self.goal = np.array([random.uniform(-0.70, 0.70), random.uniform(-0.70, 0.70), random.uniform(-0.70, 0.70)])
        
        # initialize to a random starting state and check whether it makes any sense
        thetas = []
        for joint in self.robot.joints:
             thetas.append(6.28 * np.random.random() - 3.14)
        self.robot.forwardKinmViaPositions(thetas, 1)

        if self.chill_reset == True:
            sensibility_check = False
            while not sensibility_check:
                thetas = []
                for joint in self.robot.joints:
                     thetas.append(6.28 * np.random.random() - 3.14)
                self.robot.forwardKinmViaPositions(thetas , 1)
                if calculateManipulabilityIndex(self.robot) > 0.15:
                    sensibility_check = True
        
#        for i, joint in enumerate(self.robots[robot_index].joints):
#            joint.theta = self.robots[0].joints[i].theta

# NOTE: not needed and i'm not fixing _get_obs for more robots
        obs = self._get_obs()
        return obs

    def render(self, mode='human', width=500, height=500):
        try:
            self.drawingInited == False
        except AttributeError:
            self.drawingInited = False

        if self.drawingInited == False:
            #plt.ion()
            self.fig = plt.figure()
            #self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax = self.fig.add_subplot(111, projection='3d')
            # these are for axes scaling which does not happen automatically
            self.ax.plot(np.array([0]), np.array([0]), np.array([1.5]), c='b')
            self.ax.plot(np.array([0]), np.array([0]), np.array([-1.5]), c='b')
            plt.xlim([-1.5,1.5])
            plt.ylim([-0.5,1.5])
            color_link = 'black'
            self.robot.initDrawing(self.ax, color_link)
            self.drawingInited = True
            plt.pause(0.1)
            # put this here for good measure
            self.robot.drawStateAnim()
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # manual blitting basically
        # restore region
        self.fig.canvas.restore_region(self.bg)
        # updating all the artists (set new properties (position bla))
        self.robot.drawStateAnim()
        # now you need to manually draw all the artist (otherwise you update everything)
        # thank god that's just lines, all in a list.
        # btw, that should definitely be a dictionary, not a list,
        # this makes the code fully unreadable (although correct).
        for link in self.robot.lines:
            for line in link:
                self.ax.draw_artist(line)
        self.fig.canvas.blit(self.fig.bbox)
        # NOTE: this might not work
        self.ax.set_title(str(self.n_of_tries_for_point) + 'th iteration toward goal')
        drawPoint(self.ax, self.goal, 'red', 'o')
        # if no draw it is kinda faster
        #self.fig.canvas.draw()
        # this is even faster
        #self.fig.canvas.update()
        self.fig.canvas.flush_events()


    def reset_test(self):
        self.episode_score = 0
        self.n_of_points_done += 1
        self.n_of_tries_for_point = 0

        # generate new point
        self.goal = np.array([random.uniform(-0.70, 0.70), random.uniform(-0.70, 0.70), random.uniform(-0.70, 0.70)])
        
        # DO NOT initialize to a random starting state, keep the previous one

        obs = self._get_obs()
        return obs


    def close(self):
        # close open files if any are there
        pass


    # various uitility functions COPIED from fetch_env

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _get_obs(self):
        thetas = []
        for joint in self.robot.joints:
            thetas.append(joint.theta)
        thetas = np.array(thetas , dtype=np.float32)
        obs = self.robot.p_e.copy()
        obs = np.append(obs, thetas)

        return {
            'observation': obs,
            'achieved_goal': self.robot.p_e.copy(),
            'desired_goal': self.goal.copy(),
        }


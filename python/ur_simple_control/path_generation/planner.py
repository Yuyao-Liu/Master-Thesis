from typing import List
from abc import ABC, abstractmethod

import numpy as np
from ur_simple_control.path_generation.starworlds.obstacles import StarshapedObstacle
from ur_simple_control.path_generation.starworlds.starshaped_hull import cluster_and_starify, ObstacleCluster
from ur_simple_control.path_generation.starworlds.utils.misc import tic, toc
from ur_simple_control.path_generation.star_navigation.robot.unicycle import Unicycle
from ur_simple_control.path_generation.starworlds.obstacles import StarshapedPolygon
import shapely
import yaml
import pinocchio as pin
import pickle
from importlib.resources import files

import matplotlib.pyplot as plt
import matplotlib.collections as plt_col
from multiprocessing import Queue, Lock, shared_memory

def getPlanningArgs(parser):
    robot_params_file_path = files('ur_simple_control.path_generation').joinpath('robot_params.yaml')
    tunnel_mpc_params_file_path = files('ur_simple_control.path_generation').joinpath('tunnel_mpc_params.yaml')
    parser.add_argument('--planning-robot-params-file', type=str,
                        default=robot_params_file_path,
                        #default='/home/gospodar/lund/praxis/projects/ur_simple_control/python/ur_simple_control/path_generation/robot_params.yaml',
                        #default='/home/gospodar/colcon_venv/ur_simple_control/python/ur_simple_control/path_generation/robot_params.yaml',
                        help="path to robot params file, required for path planning because it takes kinematic constraints into account")
    parser.add_argument('--tunnel-mpc-params-file', type=str,
                        default=tunnel_mpc_params_file_path,
                        #default='/home/gospodar/lund/praxis/projects/ur_simple_control/python/ur_simple_control/path_generation/tunnel_mpc_params.yaml',
                        #default='/home/gospodar/colcon_venv/ur_simple_control/python/ur_simple_control/path_generation/tunnel_mpc_params.yaml',
                        help="path to mpc (in original tunnel) params file, required for path planning because it takes kinematic constraints into account")
    parser.add_argument('--n-pol', type=int,
                        default='0',
                        help="IDK, TODO, rn this is just a preset hack value put into args for easier data transfer")
    parser.add_argument('--np', type=int,
                        default='0',
                        help="IDK, TODO, rn this is just a preset hack value put into args for easier data transfer")
    return parser

class SceneUpdater():
    def __init__(self, params: dict, verbosity=0):
        self.params = params
        self.verbosity = verbosity
        self.reset()

    def reset(self):
        self.obstacle_clusters : List[ObstacleCluster] = None
        self.free_star = None
    
    # TODO: Check why computational time varies so much for same static scene
    @staticmethod
    def workspace_modification(obstacles, p, pg, rho0, max_compute_time, hull_epsilon, gamma=0.5, make_convex=0, obstacle_clusters_prev=None, free_star_prev=None, verbosity=0):

        # Clearance variable initialization
        rho = rho0 / gamma  # First rho should be rho0
        t_init = tic()

        while True:
            if toc(t_init) > max_compute_time:
                if verbosity > 0:
                    print("[Workspace modification]: Max compute time in rho iteration.")
                break

            # Reduce rho
            rho *= gamma

            # Pad obstacles with rho
            obstacles_rho = [o.dilated_obstacle(padding=rho, id="duplicate") for o in obstacles]

            # TODO: Fix boundaries
            free_rho = shapely.geometry.box(-20, -20, 20, 20)
            for o in obstacles_rho:
                free_rho = free_rho.difference(o.polygon())

            # TODO: Check buffering fix
            # Find P0
            Bp = shapely.geometry.Point(p).buffer(0.95 * rho)
            initial_reference_set = Bp.intersection(free_rho.buffer(-0.1 * rho))

            if not initial_reference_set.is_empty:
                break

        # Initial and goal reference position selection
        r0_sh, _ = shapely.ops.nearest_points(initial_reference_set, shapely.geometry.Point(p))
        r0 = np.array(r0_sh.coords[0])
        rg_sh, _ = shapely.ops.nearest_points(free_rho, shapely.geometry.Point(pg))
        rg = np.array(rg_sh.coords[0])


        # TODO: Check more thoroughly
        if free_star_prev is not None:
            free_star_prev = free_star_prev.buffer(-1e-4)
        if free_star_prev is not None and free_star_prev.contains(r0_sh) and free_star_prev.contains(rg_sh) and free_rho.contains(free_star_prev):# not any([free_star_prev.covers(o.polygon()) for o in obstacles_rho]):
            if verbosity > 1:
                print("[Workspace modification]: Reuse workspace from previous time step.")
            obstacle_clusters = obstacle_clusters_prev
            exit_flag = 10
        else:
            # Apply cluster and starify
            obstacle_clusters, obstacle_timing, exit_flag, n_iter = \
                cluster_and_starify(obstacles_rho, r0, rg, hull_epsilon, max_compute_time=max_compute_time-toc(t_init),
                                    previous_clusters=obstacle_clusters_prev,
                                    make_convex=make_convex, verbose=verbosity)

        free_star = shapely.geometry.box(-20, -20, 20, 20)
        for o in obstacle_clusters:
            free_star = free_star.difference(o.polygon())

        compute_time = toc(t_init)
        return obstacle_clusters, r0, rg, rho, free_star, compute_time, exit_flag

    def update(self, p: np.ndarray, pg: np.ndarray, obstacles) -> tuple[np.ndarray, np.ndarray, float, list[StarshapedObstacle]]:
        # Update obstacles
        if not self.params['use_prev_workspace']:
            self.free_star = None
        self.obstacle_clusters, r0, rg, rho, self.free_star, _, _ = SceneUpdater.workspace_modification(
            obstacles, p, pg, self.params['rho0'], self.params['max_obs_compute_time'],
            self.params['hull_epsilon'], self.params['gamma'],
            make_convex=self.params['make_convex'], obstacle_clusters_prev=self.obstacle_clusters,
            free_star_prev=self.free_star, verbosity=self.verbosity)
        obstacles_star : List[StarshapedObstacle] = [o.cluster_obstacle for o in self.obstacle_clusters]
        # Make sure all polygon representations are computed
        [o._compute_polygon_representation() for o in obstacles_star]
        
        return r0, rg, rho, obstacles_star


class PathGenerator():
    def __init__(self, params: dict, verbosity=0):
        self.params = params
        self.verbosity = verbosity
        self.reset()

    def reset(self):
        self.target_path = []
    
    ### soads.py

    # TODO: Check if can make more computationally efficient
    @staticmethod
    def soads_f(r, rg, obstacles: List[StarshapedObstacle], adapt_obstacle_velocity=False, unit_magnitude=False, crep=1., reactivity=1., tail_effect=False, convergence_tolerance=1e-4, d=False):
        goal_vector = rg - r
        goal_dist = np.linalg.norm(goal_vector)
        if goal_dist < convergence_tolerance:
            return 0 * r

        No = len(obstacles)
        fa = goal_vector / goal_dist  # Attractor dynamics
        if No == 0:
            return fa

        ef = [-fa[1], fa[0]]
        Rf = np.vstack((fa, ef)).T

        mu = [obs.reference_direction(r) for obs in obstacles]
        normal = [obs.normal(r) for obs in obstacles]
        gamma = [obs.distance_function(r) for obs in obstacles]
        # Compute weights
        w = PathGenerator.compute_weights(gamma, weightPow=1)

        # Compute obstacle velocities
        xd_o = np.zeros((2, No))
        if adapt_obstacle_velocity:
            for i, obs in enumerate(obstacles):
                xd_o[:, i] = obs.vel_intertial_frame(r)

        kappa = 0.
        f_mag = 0.
        for i in range(No):
            # Compute basis matrix
            E = np.zeros((2, 2))
            E[:, 0] = mu[i]
            E[:, 1] = [-normal[i][1], normal[i][0]]
            # Compute eigenvalues
            D = np.zeros((2, 2))
            D[0, 0] = 1 - crep / (gamma[i] ** (1 / reactivity)) if tail_effect or normal[i].dot(fa) < 0. else 1
            D[1, 1] = 1 + 1 / gamma[i] ** (1 / reactivity)
            # Compute modulation
            M = E.dot(D).dot(np.linalg.inv(E))
            # f_i = M.dot(fa)
            f_i = M.dot(fa - xd_o[:, i]) + xd_o[:, i]
            # Compute contribution to velocity magnitude
            f_i_abs = np.linalg.norm(f_i)
            f_mag += w[i] * f_i_abs
            # Compute contribution to velocity direction
            nu_i = f_i / f_i_abs
            nu_i_hat = Rf.T.dot(nu_i)
            kappa_i = np.arccos(np.clip(nu_i_hat[0], -1, 1)) * np.sign(nu_i_hat[1])
            kappa += w[i] * kappa_i
        kappa_norm = abs(kappa)
        f_o = Rf.dot([np.cos(kappa_norm), np.sin(kappa_norm) / kappa_norm * kappa]) if kappa_norm > 0. else fa

        if unit_magnitude:
            f_mag = 1.
        return f_mag * f_o

    @staticmethod
    def compute_weights(
        distMeas,
        N=0,
        distMeas_lowerLimit=1,
        weightType="inverseGamma",
        weightPow=2,
    ):
        """Compute weights based on a distance measure (with no upper limit)"""
        distMeas = np.array(distMeas)
        n_points = distMeas.shape[0]

        critical_points = distMeas <= distMeas_lowerLimit

        if np.sum(critical_points):  # at least one
            if np.sum(critical_points) == 1:
                w = critical_points * 1.0
                return w
            else:
                # TODO: continuous weighting function
                # warnings.warn("Implement continuity of weighting function.")
                w = critical_points * 1.0 / np.sum(critical_points)
                return w

        distMeas = distMeas - distMeas_lowerLimit
        w = (1 / distMeas) ** weightPow
        if np.sum(w) == 0:
            return w
        w = w / np.sum(w)  # Normalization
        return w

    ### path_generator.py

    @staticmethod
    def pol2pos(path_pol, s):
        n_pol = len(path_pol) // 2 - 1
        return [sum([path_pol[j * (n_pol + 1) + i] * s ** (n_pol - i) for i in range(n_pol + 1)]) for j in range(2)]

    @staticmethod
    def path_generator(r0, rg, obstacles, dp_max, N, dt, max_compute_time, n_pol, ds_decay_rate=0.5,
                       ds_increase_rate=2., max_nr_steps=1000, convergence_tolerance=1e-5, P_prev=None, s_prev=None,
                       reactivity=1., crep=1., tail_effect=False, reference_step_size=0.5, verbosity=0):
        """
        r0 : np.ndarray                       initial reference position
        rg : np.ndarray                       goal reference position
        obstacles : list[StarshapedObstacle]  obstacles in the scene
        dp_max : float                        maximum position increment (i.e. vmax * dt)
        N : int                               ???
        dt : float                            time step
        max_compute_time : float              timeout for computation
        n_pol : int                           degree of the polynomial that fits the reference  ???
        ds_decay_rate : float                 ???
        ds_increase_rate : float              ???
        max_nr_steps : int                    computation steps threshold
        convergence_tolerance : float         ???
        P_prev : np.ndarray                   previous path  ???
        s_prev : np.ndarray                   previous path time  ???
        reactivity : float                    ???
        crep : float                          ???
        tail_effect : bool                    ???
        reference_step_size : float           ???
        verbosity : int                       you guessed it...
        """
        
        t0 = tic()

        # Initialize
        ds = 1
        s = np.zeros(max_nr_steps)
        r = np.zeros((max_nr_steps, r0.size))
        if P_prev is not None:
            i = P_prev.shape[0]
            r[:i, :] = P_prev
            s[:i] = s_prev
        else:
            i = 1
            r[0, :] = r0

        while True:
            dist_to_goal = np.linalg.norm(r[i - 1, :] - rg)
            # Check exit conditions
            if dist_to_goal < convergence_tolerance:
                if verbosity > 1:
                    print(f"[Path Generator]: Path converged. {int(100 * (s[i - 1] / N))}% of path completed.")
                break
            if s[i - 1] >= N:
                if verbosity > 1:
                    print(f"[Path Generator]: Completed path length. {int(100 * (s[i - 1] / N))}% of path completed.")
                break
            if toc(t0) > max_compute_time:
                if verbosity > 1:
                    print(f"[Path Generator]: Max compute time in path integrator. {int(100 * (s[i - 1] / N))}% of path completed.")
                break
            if i >= max_nr_steps:
                if verbosity > 1:
                    print(f"[Path Generator]: Max steps taken in path integrator. {int(100 * (s[i - 1] / N))}% of path completed.")
                break

            # Movement using SOADS dynamics
            dr = min(dp_max, dist_to_goal) * PathGenerator.soads_f(r[i - 1, :], rg, obstacles, adapt_obstacle_velocity=False,
                                                                   unit_magnitude=True, crep=crep,
                                                                   reactivity=reactivity, tail_effect=tail_effect,
                                                                   convergence_tolerance=convergence_tolerance)

            r[i, :] = r[i - 1, :] + dr * ds

            ri_in_obstacle = False
            while any([o.interior_point(r[i, :]) for o in obstacles]):
                if verbosity > 1:
                    print("[Path Generator]: Path inside obstacle. Reducing integration step from {:5f} to {:5f}.".format(ds, ds*ds_decay_rate))
                ds *= ds_decay_rate
                r[i, :] = r[i - 1, :] + dr * ds
                # Additional compute time check
                if toc(t0) > max_compute_time:
                    ri_in_obstacle = True
                    break
            if ri_in_obstacle:
                continue

            # Update travelled distance
            s[i] = s[i - 1] + ds
            # Try to increase step rate again
            ds = min(ds_increase_rate * ds, 1)
            # Increase iteration counter
            i += 1

        r = r[:i, :]
        s = s[:i]

        # Evenly spaced path
        s_vec = np.arange(0, s[-1] + reference_step_size, reference_step_size)
        xs, ys = np.interp(s_vec, s, r[:, 0]), np.interp(s_vec, s, r[:, 1])
        # Append not finished path with fixed final position
        s_vec = np.append(s_vec, np.arange(s[-1] + reference_step_size, N + reference_step_size, reference_step_size))
        xs = np.append(xs, xs[-1] * np.ones(len(s_vec)-len(xs)))
        ys = np.append(ys, ys[-1] * np.ones(len(s_vec)-len(ys)))

        reference_path = [el for p in zip(xs, ys) for el in p]  # [x0 y0 x1 y1 ...]

        # TODO: Fix when close to goal
        # TODO: Adjust for short arc length, skip higher order terms..
        path_pol = np.polyfit(s_vec, reference_path[::2], n_pol).tolist() + \
                np.polyfit(s_vec, reference_path[1::2], n_pol).tolist()  # [px0 px1 ... pxn py0 py1 ... pyn]
        # Force init position to be correct
        path_pol[n_pol] = reference_path[0]
        path_pol[-1] = reference_path[1]

        # Compute polyfit approximation error
        epsilon = [np.linalg.norm(np.array(reference_path[2 * i:2 * (i + 1)]) - np.array(PathGenerator.pol2pos(path_pol, s_vec[i]))) for i in range(N + 1)]

        compute_time = toc(t0)
        
        """
        path_pol : np.ndarray   the polynomial approximation of `reference_path`
        epsilon : [float]       approximation error between the polynomial fit and the actual path
        reference_path : list   the actual path (used for P_prev later on) in [x1, y1, x2, y2, ...] format
        compute_time : float    overall timing of this function
        """
        return path_pol, epsilon, reference_path, compute_time
        
    def prepare_prev(self, p: np.ndarray, rho: float, obstacles_star: List[StarshapedObstacle]):
        P_prev = np.array([self.target_path[::2], self.target_path[1::2]]).T
        # Shift path to start at point closest to robot position
        P_prev = P_prev[np.argmin(np.linalg.norm(p - P_prev, axis=1)):, :]
        # P_prev[0, :] = self.r0
        if np.linalg.norm(p - P_prev[0, :]) > rho:
            if self.verbosity > 0:
                print("[Path Generator]: No reuse of previous path. Path not within distance rho from robot.")
            P_prev = None
        else:
            for r in P_prev:
                if any([o.interior_point(r) for o in obstacles_star]):
                    if self.verbosity > 0:
                        print("[Path Generator]: No reuse of previous path. Path not collision-free.")
                    P_prev = None

        if P_prev is not None:
            # Cut off stand still padding in previous path
            P_prev_stepsize = np.linalg.norm(np.diff(P_prev, axis=0), axis=1)
            s_prev = np.hstack((0, np.cumsum(P_prev_stepsize) / self.params['dp_max']))
            P_prev_mask = [True] + (P_prev_stepsize > 1e-8).tolist()
            P_prev = P_prev[P_prev_mask, :]
            s_prev = s_prev[P_prev_mask]
        else:
            s_prev = None
        
        return P_prev, s_prev
    
    def update(self, p: np.ndarray, r0: np.ndarray, rg: np.ndarray, rho: float, obstacles_star: List[StarshapedObstacle]) -> tuple[List[float], float]:
        # Buffer previous target path
        if self.params['buffer'] and self.target_path:
            P_prev, s_prev = self.prepare_prev(p, rho, obstacles_star)
        else:
            P_prev, s_prev = None, None
        # Generate the new path
        path_pol, epsilon, self.target_path, _ = PathGenerator.path_generator(
            r0, rg, obstacles_star, self.params['dp_max'], self.params['N'],
            self.params['dt'], self.params['max_compute_time'], self.params['n_pol'],
            ds_decay_rate=0.5, ds_increase_rate=2., max_nr_steps=1000, P_prev=P_prev, s_prev=s_prev,
            reactivity=self.params['reactivity'], crep=self.params['crep'],
            convergence_tolerance=self.params['convergence_tolerance'], verbosity=self.verbosity)
        return path_pol, epsilon


def createMap():
    """
    createMap
    ---------
    return obstacles that define the 2D map
    """
    # [lower_left, lower_right, top_right, top_left]
    map_as_list = [
                    [[2, 2], [8, 2], [8, 3], [2, 3]]      ,
                    [[2, 3], [3, 3], [3, 4.25], [2, 4.25]],
                    [[2, 5], [8, 5], [8, 6], [2, 6]]      ,
                    [[2, 8], [8, 8], [8, 9], [2, 9]]      ,
                  ]

    obstacles = []
    for map_element in map_as_list:
        obstacles.append(StarshapedPolygon(map_element))
    return obstacles, map_as_list

def pathVisualizer(x0, goal, map_as_list, positions):
    # plotting
    fig = plt.figure()
    handle_goal = plt.plot(*pg, c="g")[0]
    handle_init = plt.plot(*x0[:2], c="b")[0]
    handle_curr = plt.plot(*x0[:2], c="r", marker=(3, 0, np.rad2deg(x0[2]-np.pi/2)), markersize=10)[0]
    handle_curr_dir = plt.plot(0, 0, marker=(2, 0, np.rad2deg(0)), markersize=5, color='w')[0]
    handle_path = plt.plot([], [], c="k")[0]
    coll = []
    for map_element in map_as_list:
        coll.append(plt_col.PolyCollection(np.array(map_element)))
    plt.gca().add_collection(coll)
    handle_title = plt.text(5, 9.5, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center")
    plt.gca().set_aspect("equal")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.draw()

    # do the updating plotting
    for x in positions:
        handle_curr.set_data([x[0]], [x[1]])
        handle_curr.set_marker((3, 0, np.rad2deg(x[2]-np.pi/2)))
        handle_curr_dir.set_data([x[0]], [x[1]])
        handle_curr_dir.set_marker((2, 0, np.rad2deg(x[2]-np.pi/2)))
        handle_path.set_data([path_gen.target_path[::2], path_gen.target_path[1::2]])
        handle_title.set_text(f"{t:5.3f}")
        fig.canvas.draw()
        plt.pause(0.005)
    plt.show()

# stupid function for stupid data re-assembly
#def path2D_to_SE2(path2D : list):
#    path2D = np.array(path2D)
#    # create (N,2) path for list [x0,y0,x1,y1,...]
#    # of course it shouldn't be like that to begin with but
#    # i have no time for that
#    path2D = path2D.reshape((-1, 2))
#    # since it's a pairwise operation it can't be done on the last point
#    x_i = path2D[:,0][:-1] # no last element
#    y_i = path2D[:,1][:-1] # no last element
#    x_i_plus_1 = path2D[:,0][1:] # no first element
#    y_i_plus_1 = path2D[:,1][1:] # no first element
#    # elementwise arctan2
#    thetas = np.arctan2(x_i_plus_1 - x_i, y_i_plus_1 - y_i)
#    thetas = thetas.reshape((-1, 1))
#    path_SE2 = np.hstack((path2D, thetas))
#    return path_SE2


def pathPointFromPathParam(n_pol, path_dim, path_pol, s):
    return [np.polyval(path_pol[j*(n_pol+1):(j+1)*(n_pol+1)], s) for j in range(path_dim)]

def path2D_timed(args, path2D_untimed, max_base_v):
    """
    path2D_timed
    ---------------------
    we have a 2D path of (N,2) shape as reference.
    it times it as this is what the current ocp formulation needs:
        there should be a timing associated with the path,
        because defining the cost function as the fit of the rollout to the path
        is complicated to do software-wise.
        as it is now, we're pre-selecting points of the path, and associating those
        with rollout states at a set time step between the states in the rollout.
        this isn't a problem if we have an idea of how fast the robot can go,
        which gives us a heuristic of how to select the points (if you went at maximum
        speed, this is how far you'd go in this time, so this is your point).
        thankfully this does not need to be exact because we just care about the distance
        between the current point and the point on the path, so if it's further out
        that just means the error is larger, and that doesn't necessarily correspond
        to a different action.
    NOTE: we are constructing a possibly bullshit
    trajectory here, it's a man-made heuristic,
    and we should leave that to the MPC,
    but that requires too much coding and there is no time rn.
    the idea is to use compute the tangent of the path,
    and use that to make a 2D frenet frame.
    this is the put to some height, making it SE3.
    i.e. roll and pitch are fixed to be 0,
    but you could make them some other constant
    """

    ####################################################
    #  getting a timed 2D trajectory from a heuristic  #
    ####################################################
    # the strategy is somewhat reasonable at least:
    # assume we're moving at 90% max velocity in the base,
    # and use that. 
    perc_of_max_v = 0.9
    base_v = perc_of_max_v * max_base_v 
    
    # 1) the length of the path divided by 0.9 * max_vel 
    #    gives us the total time of the trajectory,
    #    so we first compute that
    # easiest possible way to get approximate path length
    # (yes it should be from the polynomial approximation but that takes time to write)
    x_i = path2D_untimed[:,0][:-1] # no last element
    y_i = path2D_untimed[:,1][:-1] # no last element
    x_i_plus_1 = path2D_untimed[:,0][1:] # no first element
    y_i_plus_1 = path2D_untimed[:,1][1:] # no first element
    x_diff = x_i_plus_1 - x_i
    x_diff = x_diff.reshape((-1,1))
    y_diff = y_i_plus_1 - y_i
    y_diff = y_diff.reshape((-1,1))
    path_length = np.sum(np.linalg.norm(np.hstack((x_diff, y_diff)), axis=1))
    total_time = path_length / base_v
    # 2) we find the correspondence between s and time
    # TODO: read from where it should be, but seba checked that it's 5 for some reason
    # TODO THIS IS N IN PATH PLANNING, MAKE THIS A SHARED ARGUMENT
    max_s = 5 
    t_to_s = max_s / total_time
    # 3) construct the ocp-timed 2D path 
    # TODO MAKE REFERENCE_STEP_SIZE A SHARED ARGUMENT
    # TODO: we should know max s a priori
    reference_step_size = 0.5
    s_vec = np.arange(0, len(path2D_untimed)) / reference_step_size

    path2D = []
    # time is i * args.ocp_dt
    for i in range(args.n_knots + 1):
        # what it should be but gets stuck if we're not yet on path
        #s = (i * args.ocp_dt) * t_to_s
        # full path
        # NOTE: this should be wrong, and ocp_dt correct,
        # but it works better for some reason xd
        s = i * (max_s / args.n_knots)
        path2D.append(np.array([np.interp(s, s_vec, path2D_untimed[:,0]), 
                                np.interp(s, s_vec, path2D_untimed[:,1])]))
    path2D = np.array(path2D)
    return path2D

def path2D_to_SE3(path2D, path_height):
    """
    path2D_SE3
    ----------
    we have a 2D path of (N,2) shape as reference
    the OCP accepts SE3 (it could just rotations or translation too),
    so this function constructs it from the 2D path.
    """
    #########################
    #  path2D into pathSE2  #
    #########################
    # construct theta 
    # since it's a pairwise operation it can't be done on the last point
    x_i = path2D[:,0][:-1] # no last element
    y_i = path2D[:,1][:-1] # no last element
    x_i_plus_1 = path2D[:,0][1:] # no first element
    y_i_plus_1 = path2D[:,1][1:] # no first element
    x_diff = x_i_plus_1 - x_i
    y_diff = y_i_plus_1 - y_i
    # elementwise arctan2
    thetas = np.arctan2(x_diff, y_diff)

    ######################################
    #  construct SE3 from SE2 reference  #
    ######################################
    # the plan is parallel to the ground because it's a mobile
    # manipulation task
    pathSE3 = []
    for i in range(len(path2D) - 1):
        # first set the x axis to be in the theta direction
        # TODO: make sure this one makes sense
        rotation = np.array([
                    [-np.cos(thetas[i]), np.sin(thetas[i]), 0.0],
                    [-np.sin(thetas[i]), -np.cos(thetas[i]), 0.0],
                    [0.0,                0.0,          -1.0]])
        #rotation = pin.rpy.rpyToMatrix(0.0, 0.0, thetas[i])
        #rotation = pin.rpy.rpyToMatrix(np.pi/2, np.pi/2,0.0) @ rotation
        translation = np.array([path2D[i][0], path2D[i][1], path_height])
        pathSE3.append(pin.SE3(rotation, translation))
    pathSE3.append(pin.SE3(rotation, translation))
    return pathSE3

def path2D_to_timed_SE3(todo):
    pass

def starPlanner(goal, args, init_cmd, shm_name, lock : Lock, shm_data):
    """
    starPlanner
    ------------
    function to be put into ProcessManager,
    spitting out path points.
    it's wild dark dynamical system magic done by albin,
    with software dark magic on top to get it to spit
    out path points and just the path points.
    goal and path are [x,y],
    but there are utility functions to construct SE3 paths out of this
    elsewhere in the library.
    """
    # shm stuff
    shm = shared_memory.SharedMemory(name=shm_name)
    # dtype is default, but i have to pass it
    p_shared = np.ndarray((2,), dtype=np.float64, buffer=shm.buf)
    p = np.zeros(2)
    # environment
    obstacles, _ = createMap()    

    robot_type = "Unicycle"
    with open(args.planning_robot_params_file) as f:
        params = yaml.safe_load(f)
    robot_params = params[robot_type]
    planning_robot = Unicycle(width=robot_params['width'], 
                     vel_min=[robot_params['lin_vel_min'], -robot_params['ang_vel_max']],
                     vel_max=[robot_params['lin_vel_max'], robot_params['ang_vel_max']], 
                     name=robot_type)

    with open(args.tunnel_mpc_params_file) as f:
        params = yaml.safe_load(f)
    mpc_params = params["tunnel_mpc"]
    mpc_params['dp_max'] = planning_robot.vmax * mpc_params['dt']

    verbosity = 1
    scene_updater = SceneUpdater(mpc_params, verbosity)
    path_gen = PathGenerator(mpc_params, verbosity) 

    # TODO: make it an argument
    convergence_threshold = 0.05
    try:
        while True:
            # has to be a blocking call
            #cmd = cmd_queue.get()
            #p = cmd['p']
            # TODO: make a sendCommand type thing which will
            # handle locking, pickling or numpy-arraying for you
            # if for no other reason than not copy-pasting code
            lock.acquire()
            p[:] = p_shared[:] 
            lock.release()

            if np.linalg.norm(p - goal) < convergence_threshold:
                data_pickled = pickle.dumps("done")
                lock.acquire()
                shm_data.buf[:len(data_pickled)] = data_pickled
                #shm_data.buf[len(data_pickled):] = bytes(0)
                lock.release()
                break

            # Update the scene
            # r0 is the current position, rg is the goal, rho is a constant
            # --> why do we need to output this?
            r0, rg, rho, obstacles_star = scene_updater.update(p, goal, obstacles)
            # compute the path
            path_pol, epsilon = path_gen.update(p, r0, rg, rho, obstacles_star)
            # TODO: this is stupid, just used shared memory bro
            #if data_queue.qsize() < 1:
                #data_queue.put((path_pol, path_gen.target_path))
            data_pickled = pickle.dumps((path_pol, path_gen.target_path))
            lock.acquire()
            shm_data.buf[:len(data_pickled)] = data_pickled
            #shm_data.buf[len(data_pickled):] = bytes(0)
            lock.release()

    except KeyboardInterrupt:
        shm.close()
        shm.unlink()
        if args.debug_prints:
            print("PLANNER: caught KeyboardInterrupt, i'm out")


if __name__ == "__main__":
    pass

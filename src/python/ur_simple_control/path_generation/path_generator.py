from typing import List
from abc import ABC, abstractmethod

import numpy as np
from starworlds.obstacles import StarshapedObstacle
from starworlds.starshaped_hull import cluster_and_starify, ObstacleCluster
from starworlds.utils.misc import tic, toc
import shapely
import yaml

import matplotlib.pyplot as plt
import matplotlib.collections as plt_col



### mobile_robot.py

class MobileRobot(ABC):

    def __init__(self, nu, nx, width, name, u_min=None, u_max=None, x_min=None, x_max=None):
        """
        nx : int            number of state variables
        nu : int            number of input variables
        baseline : float    distance between the wheels
        name : str          the name of the robot
        u_min : np.ndarray  minimum input value
        u_max : np.ndarray  maximum input value
        x_min : np.ndarray  minimum state value
        x_max : np.ndarray  maximum state value
        """
        self.nx = nx
        self.nu = nu
        self.width = width
        self.name = name
        def valid_u_bound(bound): return bound is not None and len(bound) == self.nu
        def valid_q_bound(bound): return bound is not None and len(bound) == self.nx
        self.u_min = u_min if valid_u_bound(u_min) else [-np.inf] * self.nu
        self.u_max = u_max if valid_u_bound(u_max) else [np.inf] * self.nu
        self.x_min = x_min if valid_q_bound(x_min) else [-np.inf] * self.nx
        self.x_max = x_max if valid_q_bound(x_max) else [np.inf] * self.nx

    @abstractmethod
    def f(self, x, u):
        """ Forward dynamics ? """
        pass

    @abstractmethod
    def h(self, q):
        """ Forward kinematics """
        pass

    @abstractmethod
    def vel_min(self):
        pass

    @abstractmethod
    def vel_max(self):
        pass

    def move(self, x, u, dt):
        u_sat = np.clip(u, self.u_min, self.u_max)
        x_next = x + np.array(self.f(x, u_sat)) * dt
        x_next = np.clip(x_next, self.x_min, self.x_max)
        return x_next, u_sat


class Unicycle(MobileRobot):

    def __init__(self, width, vel_min=None, vel_max=None, name='robot'):
        self.vmax = vel_max[0]
        super().__init__(nu=2, nx=3, width=width, name=name, u_min=vel_min, u_max=vel_max)

    def f(self, x, u):
        return [u[0] * np.cos(x[2]),  # vx
                u[0] * np.sin(x[2]),  # vy
                u[1]]                 # wz

    def h(self, x):
        return x[:2]  # x, y

    def vel_min(self):
        return self.u_min

    def vel_max(self):
        return self.u_max

    def init_plot(self, ax=None, color='b', alpha=0.7, markersize=10):
        handles, ax = super(Unicycle, self).init_plot(ax=ax, color=color, alpha=alpha)
        handles += ax.plot(0, 0, marker=(3, 0, np.rad2deg(0)), markersize=markersize, color=color)
        handles += ax.plot(0, 0, marker=(2, 0, np.rad2deg(0)), markersize=0.5*markersize, color='w')
        return handles, ax

    def update_plot(self, x, handles):
        super(Unicycle, self).update_plot(x, handles)
        handles[1].set_data([x[0]], [x[1]])
        handles[1].set_marker((3, 0, np.rad2deg(x[2]-np.pi/2)))
        handles[1].set_markersize(handles[1].get_markersize())
        handles[2].set_data([x[0]], [x[1]])
        handles[2].set_marker((2, 0, np.rad2deg(x[2]-np.pi/2)))
        handles[2].set_markersize(handles[2].get_markersize())



### tunnel_mpc_controller.py

class SceneUpdater():
    
    def __init__(self, params: dict, verbosity=0):
        self.params = params
        self.verbosity = verbosity
        self.reset()

    def reset(self):
        self.obstacle_clusters : List[ObstacleCluster] = None
        self.free_star = None
    
    # workspace_modification.py

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


class MpcController():
    
    def __init__(self, params: dict, robot, verbosity=0):
        self.params = params
        self.robot = robot
        # self.mpc_solver = TunnelMpc(params, robot)
        self.verbosity = verbosity
        self.reset()

    def reset(self):
        # self.mpc_solver.reset()
        self.u_prev = [0] * self.robot.nu
    
    def check_convergence(self, p: np.ndarray, pg: np.ndarray):
        return np.linalg.norm(p - pg) < self.params['convergence_margin']

    def e_max(self, rho, epsilon):
        if rho > 0:
            e_max = rho - max(epsilon)
        else:
            e_max = 1.e6
        return e_max
    
    def ref(self, path_pol, s):
        n_pol = self.params['n_pol']
        return [np.polyval(path_pol[j*(n_pol+1):(j+1)*(n_pol+1)], s) for j in range(self.params['np'])]
    
    def compute_u(self, x, p, path_pol, rg, rho, epsilon):
        # Compute MPC solution
        
        # e_max = self.e_max(rho, epsilon)  # parameter for tracking error constraint
        # solution = self.mpc_solver.run(x.tolist(), self.u_prev, path_pol, self.params, e_max, rg.tolist(), self.verbosity)  # call to Rust solver
        # # Extract first control signal and store it for later
        # self.u_prev = solution['u'][:self.robot.nu]
        
        p = np.array(p)
        e_max = self.e_max(rho, epsilon)
        s_kappa = 0.9 * e_max / self.robot.vmax
        p_ref = self.ref(path_pol, s_kappa)
        
        err = p_ref - p
        dir = np.array([np.cos(x[2]), np.sin(x[2])])
        vel = 0.65 * self.robot.vmax * max(0.1, (dir @ (err / np.linalg.norm(err))))
        wel = 0.85 * ((np.arctan2(err[1], err[0]) - x[2] + np.pi) % (2*np.pi) - np.pi)
        
        alpha = 1
        self.u_prev = [alpha*vel, alpha*wel]

        return np.array(self.u_prev)


if __name__ == "__main__":
    
    from starworlds.obstacles import StarshapedPolygon
    
    # environment
    obstacles = [
        StarshapedPolygon([[2, 2], [8, 2], [8, 3], [2, 3]]),
        StarshapedPolygon([[2, 3], [3, 3], [3, 4.25], [2, 4.25]]),
        StarshapedPolygon([[2, 5], [8, 5], [8, 6], [2, 6]]),
        StarshapedPolygon([[2, 8], [8, 8], [8, 9], [2, 9]]),
    ]
    pg = np.array([0.5, 5.5])  # goal position
    p0 = np.array([9, 4])  # initial position
    theta0 = np.arctan2(pg[1]-p0[1], pg[0]-p0[0])  # initial heading (simply start heading towards goal)
    
    # robot
    robot_type = "Unicycle"
    with open(r'robot_params.yaml') as f:
        params = yaml.safe_load(f)
    robot_params = params[robot_type]
    robot = Unicycle(width=robot_params['width'], 
                     vel_min=[robot_params['lin_vel_min'], -robot_params['ang_vel_max']],
                     vel_max=[robot_params['lin_vel_max'], robot_params['ang_vel_max']], 
                     name=robot_type)
    x0 = np.append(p0, [theta0])  # initial robot state
    
    # MPC
    with open(r'./tunnel_mpc_params.yaml') as f:
        params = yaml.safe_load(f)
    mpc_params = params["tunnel_mpc"]
    mpc_params['dp_max'] = robot.vmax * mpc_params['dt']
    
    # components
    verbosity = 1
    scene_updater = SceneUpdater(mpc_params, verbosity)
    path_gen = PathGenerator(mpc_params, verbosity) 
    controller = MpcController(mpc_params, robot, verbosity)
    
    
    # plotting
    fig = plt.figure()
    handle_goal = plt.plot(*pg, c="g")[0]
    handle_init = plt.plot(*x0[:2], c="b")[0]
    handle_curr = plt.plot(*x0[:2], c="r", marker=(3, 0, np.rad2deg(x0[2]-np.pi/2)), markersize=10)[0]
    handle_curr_dir = plt.plot(0, 0, marker=(2, 0, np.rad2deg(0)), markersize=5, color='w')[0]
    handle_path = plt.plot([], [], c="k")[0]
    coll = plt_col.PolyCollection([
        np.array([[2, 2], [8, 2], [8, 3], [2, 3]]),
        np.array([[2, 3], [3, 3], [3, 4.25], [2, 4.25]]),
        np.array([[2, 5], [8, 5], [8, 6], [2, 6]]),
        np.array([[2, 8], [8, 8], [8, 9], [2, 9]])
    ])
    plt.gca().add_collection(coll)
    handle_title = plt.text(5, 9.5, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center")
    plt.gca().set_aspect("equal")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.draw()
    
    
    # run the controller
    T_max = 30
    dt = controller.params['dt']
    t = 0.
    x = x0
    u_prev = np.zeros(robot.nu)
    convergence_threshold = 0.05
    converged = False
    try:
        while t < T_max:
            p = robot.h(x)
            
            if np.linalg.norm(p - pg) < convergence_threshold:
                break
            
            # Udpate the scene
            r0, rg, rho, obstacles_star = scene_updater.update(p, pg, obstacles)

            # Check for convergence
            if controller.check_convergence(p, pg):
                u = np.zeros(robot.nu)
            else:
                # Update target path
                path_pol, epsilon = path_gen.update(p, r0, rg, rho, obstacles_star)
                # Calculate next control input
                u = controller.compute_u(x, p, path_pol, rg, rho, epsilon)
            
            # update robot position
            x, _ = robot.move(x, u, dt)
            u_prev = u
            t += dt
            
            # plot
            handle_curr.set_data([x[0]], [x[1]])
            handle_curr.set_marker((3, 0, np.rad2deg(x[2]-np.pi/2)))
            handle_curr_dir.set_data([x[0]], [x[1]])
            handle_curr_dir.set_marker((2, 0, np.rad2deg(x[2]-np.pi/2)))
            handle_path.set_data([path_gen.target_path[::2], path_gen.target_path[1::2]])
            handle_title.set_text(f"{t:5.3f}")
            fig.canvas.draw()
            plt.pause(0.005)
        
    except Exception as ex:
        raise ex
        pass
    
    plt.show()
    

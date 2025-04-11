from smc.path_generation.starworlds.obstacles import StarshapedObstacle
from smc.path_generation.starworlds.utils.misc import tic, toc

import numpy as np
from typing import List

class PathGenerator:
    def __init__(self, params: dict, verbosity=0):
        self.params = params
        self.verbosity = verbosity
        self.reset()

    def reset(self):
        self.target_path = []

    ### soads.py

    # TODO: Check if can make more computationally efficient
    @staticmethod
    def soads_f(
        r,
        rg,
        obstacles: List[StarshapedObstacle],
        adapt_obstacle_velocity=False,
        unit_magnitude=False,
        crep=1.0,
        reactivity=1.0,
        tail_effect=False,
        convergence_tolerance=1e-4,
        d=False,
    ):
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

        kappa = 0.0
        f_mag = 0.0
        for i in range(No):
            # Compute basis matrix
            E = np.zeros((2, 2))
            E[:, 0] = mu[i]
            E[:, 1] = [-normal[i][1], normal[i][0]]
            # Compute eigenvalues
            D = np.zeros((2, 2))
            D[0, 0] = (
                1 - crep / (gamma[i] ** (1 / reactivity))
                if tail_effect or normal[i].dot(fa) < 0.0
                else 1
            )
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
        f_o = (
            Rf.dot([np.cos(kappa_norm), np.sin(kappa_norm) / kappa_norm * kappa])
            if kappa_norm > 0.0
            else fa
        )

        if unit_magnitude:
            f_mag = 1.0
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
        return [
            sum(
                [
                    path_pol[j * (n_pol + 1) + i] * s ** (n_pol - i)
                    for i in range(n_pol + 1)
                ]
            )
            for j in range(2)
        ]

    @staticmethod
    def path_generator(
        r0,
        rg,
        obstacles,
        dp_max,
        N,
        dt,
        max_compute_time,
        n_pol,
        ds_decay_rate=0.5,
        ds_increase_rate=2.0,
        max_nr_steps=1000,
        convergence_tolerance=1e-5,
        P_prev=None,
        s_prev=None,
        reactivity=1.0,
        crep=1.0,
        tail_effect=False,
        reference_step_size=0.5,
        verbosity=0,
    ):
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
                    print(
                        f"[Path Generator]: Path converged. {int(100 * (s[i - 1] / N))}% of path completed."
                    )
                break
            if s[i - 1] >= N:
                if verbosity > 1:
                    print(
                        f"[Path Generator]: Completed path length. {int(100 * (s[i - 1] / N))}% of path completed."
                    )
                break
            if toc(t0) > max_compute_time:
                if verbosity > 1:
                    print(
                        f"[Path Generator]: Max compute time in path integrator. {int(100 * (s[i - 1] / N))}% of path completed."
                    )
                break
            if i >= max_nr_steps:
                if verbosity > 1:
                    print(
                        f"[Path Generator]: Max steps taken in path integrator. {int(100 * (s[i - 1] / N))}% of path completed."
                    )
                break

            # Movement using SOADS dynamics
            dr = min(dp_max, dist_to_goal) * PathGenerator.soads_f(
                r[i - 1, :],
                rg,
                obstacles,
                adapt_obstacle_velocity=False,
                unit_magnitude=True,
                crep=crep,
                reactivity=reactivity,
                tail_effect=tail_effect,
                convergence_tolerance=convergence_tolerance,
            )

            r[i, :] = r[i - 1, :] + dr * ds

            ri_in_obstacle = False
            while any([o.interior_point(r[i, :]) for o in obstacles]):
                if verbosity > 1:
                    print(
                        "[Path Generator]: Path inside obstacle. Reducing integration step from {:5f} to {:5f}.".format(
                            ds, ds * ds_decay_rate
                        )
                    )
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
        s_vec = np.append(
            s_vec,
            np.arange(
                s[-1] + reference_step_size,
                N + reference_step_size,
                reference_step_size,
            ),
        )
        xs = np.append(xs, xs[-1] * np.ones(len(s_vec) - len(xs)))
        ys = np.append(ys, ys[-1] * np.ones(len(s_vec) - len(ys)))

        reference_path = [el for p in zip(xs, ys) for el in p]  # [x0 y0 x1 y1 ...]

        # TODO: Fix when close to goal
        # TODO: Adjust for short arc length, skip higher order terms..
        path_pol = (
            np.polyfit(s_vec, reference_path[::2], n_pol).tolist()
            + np.polyfit(s_vec, reference_path[1::2], n_pol).tolist()
        )  # [px0 px1 ... pxn py0 py1 ... pyn]
        # Force init position to be correct
        path_pol[n_pol] = reference_path[0]
        path_pol[-1] = reference_path[1]

        # Compute polyfit approximation error
        epsilon = [
            np.linalg.norm(
                np.array(reference_path[2 * i : 2 * (i + 1)])
                - np.array(PathGenerator.pol2pos(path_pol, s_vec[i]))
            )
            for i in range(N + 1)
        ]

        compute_time = toc(t0)

        """
        path_pol : np.ndarray   the polynomial approximation of `reference_path`
        epsilon : [float]       approximation error between the polynomial fit and the actual path
        reference_path : list   the actual path (used for P_prev later on) in [x1, y1, x2, y2, ...] format
        compute_time : float    overall timing of this function
        """
        return path_pol, epsilon, reference_path, compute_time

    def prepare_prev(
        self, p: np.ndarray, rho: float, obstacles_star: List[StarshapedObstacle]
    ):
        P_prev = np.array([self.target_path[::2], self.target_path[1::2]]).T
        # Shift path to start at point closest to robot position
        P_prev = P_prev[np.argmin(np.linalg.norm(p - P_prev, axis=1)) :, :]
        # P_prev[0, :] = self.r0
        if np.linalg.norm(p - P_prev[0, :]) > rho:
            if self.verbosity > 0:
                print(
                    "[Path Generator]: No reuse of previous path. Path not within distance rho from robot."
                )
            P_prev = None
        else:
            for r in P_prev:
                if any([o.interior_point(r) for o in obstacles_star]):
                    if self.verbosity > 0:
                        print(
                            "[Path Generator]: No reuse of previous path. Path not collision-free."
                        )
                    P_prev = None

        if P_prev is not None:
            # Cut off stand still padding in previous path
            P_prev_stepsize = np.linalg.norm(np.diff(P_prev, axis=0), axis=1)
            s_prev = np.hstack((0, np.cumsum(P_prev_stepsize) / self.params["dp_max"]))
            P_prev_mask = [True] + (P_prev_stepsize > 1e-8).tolist()
            P_prev = P_prev[P_prev_mask, :]
            s_prev = s_prev[P_prev_mask]
        else:
            s_prev = None

        return P_prev, s_prev

    def update(
        self,
        p: np.ndarray,
        r0: np.ndarray,
        rg: np.ndarray,
        rho: float,
        obstacles_star: List[StarshapedObstacle],
    ) -> tuple[List[float], float]:
        # Buffer previous target path
        if self.params["buffer"] and self.target_path:
            P_prev, s_prev = self.prepare_prev(p, rho, obstacles_star)
        else:
            P_prev, s_prev = None, None
        # Generate the new path
        path_pol, epsilon, self.target_path, _ = PathGenerator.path_generator(
            r0,
            rg,
            obstacles_star,
            self.params["dp_max"],
            self.params["N"],
            self.params["dt"],
            self.params["max_compute_time"],
            self.params["n_pol"],
            ds_decay_rate=0.5,
            ds_increase_rate=2.0,
            max_nr_steps=1000,
            P_prev=P_prev,
            s_prev=s_prev,
            reactivity=self.params["reactivity"],
            crep=self.params["crep"],
            convergence_tolerance=self.params["convergence_tolerance"],
            verbosity=self.verbosity,
        )
        return path_pol, epsilon

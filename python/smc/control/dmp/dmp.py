from smc.robots.abstract_robotmanager import AbstractRobotManager
from smc.control.control_loop_manager import ControlLoopManager

import numpy as np
import argparse
from functools import partial
from collections import deque

# TODO:
# 1. change the dimensions so that they make sense,
#    i.e. shape = (N_points, dimension_of_points)
# 2. change hand-written numerical differentiation
#    to numpy calls (for code style more than anything)
# 3. separate x into z and s variables, this is unnecessarily complicated
# 4. ask mentors if there's ever a reason not to use temporal coupling,
#    and if not, integrate it into the DMP class

# k,d are constanst which determine the baseline uncostrained dynamics
# these work fine, but it could be good to play around with them just to
# see their effect. normally people set them so that you get critical damping
# as the uncostrained system


def getDMPArgs(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    #############################
    #  dmp  specific arguments  #
    #############################
    parser.add_argument(
        "--temporal-coupling",
        action=argparse.BooleanOptionalAction,
        help="whether you want to use temporal coupling",
        default=True,
    )
    parser.add_argument(
        "--tau0",
        type=float,
        help="total time needed for trajectory. if you use temporal coupling,\
                  you can still follow the path even if it's too fast",
        default=10,
    )
    parser.add_argument(
        "--gamma-nominal",
        type=float,
        help="positive constant for tuning temporal coupling: the higher,\
            the fast the return rate to nominal tau",
        default=1.0,
    )
    parser.add_argument(
        "--gamma-a",
        type=float,
        help="positive constant for tuning temporal coupling, potential term",
        default=0.5,
    )
    parser.add_argument(
        "--eps-tc",
        type=float,
        help="temporal coupling term, should be small",
        default=0.001,
    )
    # default=0.05)
    return parser


class DMP:
    def __init__(self, trajectory: str | np.ndarray, k=100, d=20, a_s:float=1.0, n_bfs=100):
        # TODO load the trajectory here
        # and then allocate all these with zeros of appopriate length
        # as this way they're basically declared twice

        # parameters
        # for whatever reason, k, d and a_s don't match the ,
        # OG formulation: tau * z_dot = alpha_z * (beta_z * (g - y) - z) + f(x)
        # this formulation: tau * z_dot = k * (g - y) - d * z + f(x) = h (z, y, s)
        # i.e. k = beta_z * alpha_z
        #      d = alpha_z
        self.k: int = k
        self.d: int = d
        self.a_s = a_s
        self.n_bfs: int = n_bfs

        # trajectory parameters
        self.n = 0  # n of dofs
        self.y0: np.ndarray = None  # initial position
        self.tau0: float = None  # final time
        self.g: np.ndarray = None  # goal positions
        # scaling factor, updated online to ensure traj is followed
        self.tau: float = None

        # initialize basis functions for LWR
        self.w = None  # weights of basis functions
        self.centers = None  # centers of basis functions
        self.widths = None  # widths of basis functions

        # state
        self.x = None
        self.theta = None
        self.pos: np.ndarray = None  # position
        self.vel: np.ndarray = None  # velocity
        self.acc: np.ndarray = None  # acceleration

        # desired path
        self.path = None

        # actually init
        # TODO handle this better, this is not optimal programming
        if type(trajectory) == str:
            self.load_trajectory_from_file(trajectory)
        else:
            self.load_trajectory(trajectory)
        self.fit()

    def load_trajectory_from_file(self, file_path):
        # load trajectory.  this is just joint positions.
        trajectory = np.genfromtxt(file_path, delimiter=",")
        self.time = trajectory[:, 0]
        self.time = self.time.reshape(1, len(self.time))
        self.y = np.array(trajectory[:, 1:]).T

    def load_trajectory(self, trajectory: np.ndarray):
        # load trajectory.  this is just joint positions.
        self.time = trajectory[:, 0]
        self.time = self.time.reshape(1, len(self.time))
        self.y = np.array(trajectory[:, 1:]).T

    def reset(self):
        self.x = np.vstack((np.zeros((self.n, 1)), self.y0, 1.0))
        self.tau = self.tau0
        self.theta = 0
        self.pos = self.y0
        self.vel = 0 * self.y0
        self.acc = 0 * self.y0

    def z(self, x=None):
        if x is None:
            x = self.x
        return x[0 : self.n]

    def y_fun(self, x=None):
        if x is None:
            x = self.x
        return x[self.n : 2 * self.n]

    def s(self, x=None):
        if x is None:
            x = self.x
        return x[2 * self.n]

    def set_goal(self, g):
        self.g = g

    def set_tau(self, tau):
        self.tau = tau

    def psi(self, s, i=None):
        if i is not None and len(s) == 1:
            return np.exp(-1 / (2 * self.widths[i] ** 2) * (s - self.centers[i]) ** 2)
        if i is None:
            return np.exp(-1 / (2 * self.widths**2) * (s - self.centers) ** 2)

    def h(self, x=None):
        if x is None:
            x = self.x
        psi = self.psi(self.s(x)).reshape((self.n_bfs, 1))
        v = (
            (self.w.dot(psi))
            / np.maximum(np.sum(psi), 1e-8)
            * (self.g - self.y0)
            * self.s(x)
        )
        h = self.k * (self.g - self.y_fun(x)) - self.d * self.z(x) + v
        return h

    def f(self, x=None):
        if x is None:
            x = self.x
        return np.vstack((self.h(x), self.z(x), -self.a_s * self.s(x)))

    def step(self, dt):
        # Update state
        self.x = self.x + self.f() / self.tau * dt
        self.theta = self.theta + 1 / self.tau * dt

        # Extract trajectory state
        vel_prev = self.vel
        self.pos = self.y_fun()
        self.vel = self.z() / self.tau
        self.acc = (self.vel - vel_prev) / dt

    # if you don't know what the letters mean,
    # look at the equation (which you need to know anyway)
    def fit(self):
        # Set target trajectory parameters
        self.n = self.y.shape[0]
        self.y0 = self.y[:, 0].reshape((self.n, 1))
        self.g = self.y[:, -1].reshape((self.n, 1))
        self.tau0 = self.time[0, -1]

        # Set basis functions
        t_centers = np.linspace(0, self.tau0, self.n_bfs, endpoint=True)
        self.centers = np.exp(-self.a_s / self.tau0 * t_centers)
        widths = np.abs((np.diff(self.centers)))
        self.widths = np.concatenate((widths, [widths[-1]]))

        # Calculate derivatives
        yd_demo = (self.y[:, 1:] - self.y[:, :-1]) / (
            self.time[0, 1:] - self.time[0, :-1]
        )
        yd_demo = np.concatenate((yd_demo, np.zeros((self.n, 1))), axis=1)
        ydd_demo = (yd_demo[:, 1:] - yd_demo[:, :-1]) / (
            self.time[0, 1:] - self.time[0, :-1]
        )
        ydd_demo = np.concatenate((ydd_demo, np.zeros((self.n, 1))), axis=1)

        # Compute weights
        s_seq = np.exp(-self.a_s / self.tau0 * self.time)
        self.w = np.zeros((self.n, self.n_bfs))
        for i in range(self.n):
            if abs(self.g[i] - self.y0[i]) < 1e-5:
                continue
            f_gain = s_seq * (self.g[i] - self.y0[i])
            f_target = (
                self.tau0**2 * ydd_demo[i, :]
                - self.k * (self.g[i] - self.y[i, :])
                + self.d * self.tau0 * yd_demo[i, :]
            )
            for j in range(self.n_bfs):
                psi_j = self.psi(s_seq, j)
                num = f_gain.dot((psi_j * f_target).T)
                den = f_gain.dot((psi_j * f_gain).T)
                if abs(den) < 1e-6:
                    continue
                self.w[i, j] = num / den

        # Reset state
        self.reset()


class NoTC:
    def update(self, dmp, dt):
        return 0


class TCVelAccConstrained:

    def __init__(self, gamma_nominal, gamma_a, v_max, a_max, eps=0.001):
        self.gamma_nominal = gamma_nominal
        self.gamma_a = gamma_a
        self.eps = eps
        self.v_max = v_max.reshape((len(v_max), 1))
        self.a_max = a_max.reshape((len(a_max), 1))

    def generate_matrices(self, dmp, dt):
        A = np.vstack((-dmp.z(), dmp.z()))
        B = np.vstack((-self.a_max, -self.a_max))
        C = np.vstack((dmp.h(), -dmp.h()))
        D = np.vstack((-self.v_max, -self.v_max))
        x_next = dmp.x + dmp.f(dmp.x) / dmp.tau * dt
        A_next = np.vstack((-dmp.z(x_next), dmp.z(x_next)))
        C_next = np.vstack((dmp.h(x_next), -dmp.h(x_next)))
        return A, B, C, D, A_next, C_next

    def update(self, dmp, dt):

        A, B, C, D, A_next, C_next = self.generate_matrices(dmp, dt)

        # Acceleration bounds
        i = np.squeeze(A < 0)
        if i.any():
            taud_min_a = np.max(-(B[i] * dmp.tau**2 + C[i]) / A[i])
        else:
            taud_min_a = -np.inf
        i = np.squeeze(A > 0)
        if i.any():
            taud_max_a = np.min(-(B[i] * dmp.tau**2 + C[i]) / A[i])
        else:
            taud_max_a = np.inf
        # Velocity bounds
        i = range(len(A_next))
        tau_min_v = np.max(-A_next[i] / D[i])
        taud_min_v = (tau_min_v - dmp.tau) / dt
        # Feasibility bounds
        ii = np.arange(len(A_next))[np.squeeze(A_next < 0)]
        jj = np.arange(len(A_next))[np.squeeze(A_next > 0)]
        tau_min_f = -np.inf
        for i in ii:
            for j in jj:
                num = C_next[i] * abs(A_next[j]) + C_next[j] * abs(A_next[i])
                if num > 0:
                    den = abs(B[i] * A_next[j]) + abs(B[j] * A_next[i])
                    tmp = np.sqrt(num / den)
                    if tmp > tau_min_f:
                        tau_min_f = tmp
        taud_min_f = (tau_min_f - dmp.tau) / dt
        # Nominal bound
        taud_min_nominal = (dmp.tau0 - dmp.tau) / dt

        taud_min = np.max((taud_min_a, taud_min_v, taud_min_f[0], taud_min_nominal))

        # Base update law
        ydd_bar = dmp.h() / (dmp.tau**2 * self.a_max)
        if self.gamma_a > 0:
            pot_a = self.gamma_a * np.sum(
                ydd_bar**2
                / np.maximum(
                    1 - ydd_bar**2, self.gamma_a * self.eps * np.ones((len(ydd_bar), 1))
                )
            )
        else:
            pot_a = 0
        # pot_a = self.gamma_a * np.amax(ydd_bar ** 2 / np.maximum(1 - ydd_bar ** 2, self.gamma_a * self.eps * np.ones((len(ydd_bar), 1))))
        taud = self.gamma_nominal * (dmp.tau0 - dmp.tau) + dmp.tau * pot_a

        # Saturate
        taud = np.min((taud, taud_max_a))
        taud = np.max((taud, taud_min))

        return taud


def controlLoopDMP(
    args: argparse.Namespace,
    robot: AbstractRobotManager,
    dmp: DMP,
    tc: NoTC | TCVelAccConstrained,
    i: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[bool, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopDMP
    -----------------------------
    execute a set-up dmp
    """
    breakFlag = False
    log_item = {}
    save_past_dict = {}

    q = robot.q

    dmp.step(robot.dt)
    tau_dmp = dmp.tau + tc.update(dmp, robot.dt) * robot.dt
    dmp.set_tau(tau_dmp)

    vel_cmd = dmp.vel + args.kp * (dmp.pos - q.reshape((-1, 1)))

    robot.sendVelocityCommand(vel_cmd)

    if (np.linalg.norm(dmp.vel) < 0.01) and (i > int(dmp.tau0 * 500)):
        breakFlag = True

    log_item["qs"] = q.reshape((robot.model.nq,))
    log_item["dmp_qs"] = dmp.pos.reshape((6,))
    log_item["dqs"] = robot.v
    log_item["dmp_dqs"] = dmp.vel.reshape((6,))
    return breakFlag, save_past_dict, log_item


def followDMP(
    args: argparse.Namespace, robot: AbstractRobotManager, qs: np.ndarray, tau0: float
) -> None:
    t = np.linspace(0, tau0, len(qs)).reshape((len(qs), 1))
    joint_trajectory = np.hstack((t, qs))
    dmp = DMP(joint_trajectory)
    if not args.temporal_coupling:
        tc = NoTC()
    else:
        v_max_ndarray = np.ones(robot.nq) * robot._max_v
        a_max_ndarray = np.ones(robot.nq) * args.acceleration
        tc = TCVelAccConstrained(
            args.gamma_nominal, args.gamma_a, v_max_ndarray, a_max_ndarray, args.eps_tc
        )
    save_past_dict = {}
    log_item = {}
    log_item["qs"] = np.zeros((robot.model.nq,))
    log_item["dmp_qs"] = np.zeros((6,))
    log_item["dqs"] = np.zeros((robot.model.nv,))
    log_item["dmp_dqs"] = np.zeros((6,))
    controlLoop = partial(controlLoopDMP, args, robot, dmp, tc)
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    loop_manager.run()

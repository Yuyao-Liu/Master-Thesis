import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import argparse
from functools import partial
from ur_simple_control.managers import getMinimalArgParser, ControlLoopManager, RobotManager

def get_args():
    parser = getMinimalArgParser()
    parser.description = 'trying to get 3dcontact to do something'
    # add more arguments here from different Simple Manipulator Control modules
    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = get_args()
    robot = RobotManager(args)

    # i'm not using a control loop because i'm copy pasting code
    # so let's go pinocchio all the way
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    q0 = pin.neutral(robot.model)
    q0 = pin.randomConfiguration(robot.model)
    viz.display(q0)

    ee_constraint_placement = pin.SE3.Identity()
    #ee_constraint_placement.translation = np.array([0.3,0.3,0.3])
    ee_constraint_placement.translation = np.array([0.0,0.0,0.5])

    # TODO
    # 1. TODO
    #    try the thing with 2 joints (same class pin.RigidConstraintModel)
    #    to connect cart and hand
    # 2. TODO
    #    figure  out what the reference frame argument here means
    # 3. TODO
    #    figure out what different reference frames do in this context 
    constraint_model = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_3D,
        robot.model,
        robot.JOINT_ID,
        #pin.ReferenceFrame.LOCAL # but i have no idea what this is really (is it in ee frame?)
        ee_constraint_placement # but i have no idea what this is really (is it in ee frame?)
    )
    constraint_data = constraint_model.createData()
    constraint_dim = constraint_model.size()

    # some inverse geometry for some reason unknown to me
    rho = 1e-10 # what is this
    mu = 1e-4  # what is this
    eps = 1e-10
    y = np.ones((constraint_dim))
    robot.data.M = np.eye(robot.model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(robot.model, [constraint_model])
    # what is this?
    # i guess it's internal calculation for constrained dynamics,
    # but no idea
    # maybe we need a feasible initial position
    q = pin.randomConfiguration(robot.model)
    for i in range(1000):
        pin.computeJointJacobians(robot.model, robot.data, q)
        kkt_constraint.compute(robot.model, robot.data, [constraint_model], [constraint_data], mu)
        # i have no idea what this is
        # is this a constraint force?
        constraint_value = constraint_data.c1Mc2.translation
        J = pin.getFrameJacobian(robot.model, robot.data, constraint_model.joint1_id, constraint_model.joint1_placement, constraint_model.reference_frame)[:3, :] # why cut off?
        primal_feas = np.linalg.norm(constraint_value, np.inf) # why inf norm?
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps and dual_feas < eps:
            print("converged")
            break
        rhs = np.concatenate([-constraint_value - y * mu, np.zeros(robot.model.nv)])
        # i'm assuming these shenanegans are here due to constraint_dim < 6
        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 1.0
        q = pin.integrate(robot.model, q, -alpha * dq)
        y -= alpha * (-dy + y)

    v = np.zeros(robot.model.nv)
    tau = np.zeros(robot.model.nv)
    dt = 5e-3
    T_sim = 1000
    t = 0
    mu_sim = 1e-10
    constraint_model.corrector.Kp[:] = 10
    constraint_model.corrector.Kd[:] = 2.0 * np.sqrt(constraint_model.corrector.Kp)
    pin.initConstraintDynamics(robot.model, robot.data, [constraint_model])
    prox_settings = pin.ProximalSettings(1e-8, mu_sim, 10)


    while t <= T_sim:
        a = pin.constraintDynamics(
            robot.model, robot.data, q, v, tau, [constraint_model], [constraint_data], prox_settings
        )
        v += a * dt
        q = pin.integrate(robot.model, q, v * dt)
        viz.display(q)
        time.sleep(dt)
        t += dt

    # get expected behaviour here (library can't know what the end is - you have to do this here)
    if not args.pinocchio_only:
        robot.stopRobot()

    if args.save_log:
        robot.log_manager.plotAllControlLoops()

    if args.visualize_manipulator:
        robot.killManipulatorVisualizer()
    
    if args.save_log:
        robot.log_manager.saveLog()
    #loop_manager.stopHandler(None, None)

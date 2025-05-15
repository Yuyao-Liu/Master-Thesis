"""
possible improvement: 
    get the calibration file of the robot without ros
    and then put it here.
    these magic numbers are not a good look.
"""

import pinocchio as pin
import numpy as np
import sys
import os
from importlib.resources import files
import hppfcl as fcl
import example_robot_data
import crocoddyl
# can't get the urdf reading with these functions to save my life, idk what or why

#############################################################
# PACKAGE_DIR IS THE WHOLE UR_SIMPLE_CONTROL FOLDER (cos that's accessible from anywhere it's installed)
# PACKAGE:// IS WHAT'S BEING REPLACED WITH THE PACKAGE_DIR ARGUMENT IN THE URDF.
# YOU GIVE ABSOLUTE PATH TO THE URDF THO.
#############################################################

"""
loads what needs to be loaded.
calibration for the particular robot was extracted from the yml
obtained from ros and appears here as magic numbers.
i have no idea how to extract calibration data without ros
and i have no plans to do so.
aligning what UR thinks is the world frame
and what we think is the world frame is not really necessary,
but it does aleviate some brain capacity while debugging.
having that said, supposedly there is a big missalignment (few cm)
between the actual robot and the non-calibrated model.
NOTE: this should be fixed for a proper release
"""


def get_model():

    urdf_path_relative = files("ur_simple_control.robot_descriptions.urdf").joinpath(
        "ur5e_with_robotiq_hande_FIXED_PATHS.urdf"
    )
    # urdf_path_relative = files("ur_simple_control.robot_descriptions.urdf").joinpath(
    #     "model.urdf"
    # )
    urdf_path_absolute = os.path.abspath(urdf_path_relative)
    mesh_dir = files("ur_simple_control")
    mesh_dir_absolute = os.path.abspath(mesh_dir)

    shoulder_trans = np.array([0, 0, 0.1625134425523304])
    shoulder_rpy = np.array([-0, 0, 5.315711138647629e-08])
    shoulder_se3 = pin.SE3(pin.rpy.rpyToMatrix(shoulder_rpy), shoulder_trans)

    upper_arm_trans = np.array([0.000300915150907851, 0, 0])
    upper_arm_rpy = np.array([1.571659987714477, 0, 1.155342090832558e-06])
    upper_arm_se3 = pin.SE3(pin.rpy.rpyToMatrix(upper_arm_rpy), upper_arm_trans)

    forearm_trans = np.array([-0.4249536100418752, 0, 0])
    forearm_rpy = np.array([3.140858652067472, 3.141065383898231, 3.141581851193229])
    forearm_se3 = pin.SE3(pin.rpy.rpyToMatrix(forearm_rpy), forearm_trans)

    wrist_1_trans = np.array(
        [-0.3922353894477613, -0.001171506236920081, 0.1337997346972175]
    )
    wrist_1_rpy = np.array(
        [0.008755445624588536, 0.0002860523431017214, 7.215921353974553e-06]
    )
    wrist_1_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_1_rpy), wrist_1_trans)

    wrist_2_trans = np.array(
        [5.620166987673597e-05, -0.09948910981796041, 0.0002201494606859632]
    )
    wrist_2_rpy = np.array([1.568583530823855, 0, -3.513049549874747e-07])
    wrist_2_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_2_rpy), wrist_2_trans)

    wrist_3_trans = np.array(
        [9.062061300900664e-06, 0.09947787349620175, 0.0001411778743239612]
    )
    wrist_3_rpy = np.array([1.572215514545703, 3.141592653589793, 3.141592633687631])
    wrist_3_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_3_rpy), wrist_3_trans)

    model = None
    collision_model = None
    visual_model = None
    # this command just calls the ones below it. both are kept here
    # in case pinocchio people decide to change their api.
    # model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path_absolute, mesh_dir_absolute)
    model = pin.buildModelFromUrdf(urdf_path_absolute)
    visual_model = pin.buildGeomFromUrdf(
        model, urdf_path_absolute, pin.GeometryType.VISUAL, None, mesh_dir_absolute
    )
    collision_model = pin.buildGeomFromUrdf(
        model, urdf_path_absolute, pin.GeometryType.COLLISION, None, mesh_dir_absolute
    )

    # for whatever reason the hand-e files don't have/
    # meshcat can't read scaling information.
    # so we scale manually,
    # and the stupid gripper is in milimeters
    for geom in visual_model.geometryObjects:
        if "hand" in geom.name:
            s = geom.meshScale
            # this looks exactly correct lmao
            s *= 0.001
            geom.meshScale = s
    for geom in collision_model.geometryObjects:
        if "hand" in geom.name:
            s = geom.meshScale
            # this looks exactly correct lmao
            s *= 0.001
            geom.meshScale = s

    # updating joint placements.
    model.jointPlacements[1] = shoulder_se3
    model.jointPlacements[2] = upper_arm_se3
    model.jointPlacements[3] = forearm_se3
    model.jointPlacements[4] = wrist_1_se3
    model.jointPlacements[5] = wrist_2_se3
    model.jointPlacements[6] = wrist_3_se3
    # TODO: fix where the fingers end up by setting a better position here (or maybe not here idk)
    #model = pin.buildReducedModel(model, [7, 8], np.zeros(model.nq))
    data = pin.Data(model)

    return model, collision_model, visual_model, data


def getGripperlessUR5e():
    robot = example_robot_data.load("ur5")

    shoulder_trans = np.array([0, 0, 0.1625134425523304])
    shoulder_rpy = np.array([-0, 0, 5.315711138647629e-08])
    shoulder_se3 = pin.SE3(pin.rpy.rpyToMatrix(shoulder_rpy), shoulder_trans)

    upper_arm_trans = np.array([0.000300915150907851, 0, 0])
    upper_arm_rpy = np.array([1.571659987714477, 0, 1.155342090832558e-06])
    upper_arm_se3 = pin.SE3(pin.rpy.rpyToMatrix(upper_arm_rpy), upper_arm_trans)

    forearm_trans = np.array([-0.4249536100418752, 0, 0])
    forearm_rpy = np.array([3.140858652067472, 3.141065383898231, 3.141581851193229])
    forearm_se3 = pin.SE3(pin.rpy.rpyToMatrix(forearm_rpy), forearm_trans)

    wrist_1_trans = np.array(
        [-0.3922353894477613, -0.001171506236920081, 0.1337997346972175]
    )
    wrist_1_rpy = np.array(
        [0.008755445624588536, 0.0002860523431017214, 7.215921353974553e-06]
    )
    wrist_1_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_1_rpy), wrist_1_trans)

    wrist_2_trans = np.array(
        [5.620166987673597e-05, -0.09948910981796041, 0.0002201494606859632]
    )
    wrist_2_rpy = np.array([1.568583530823855, 0, -3.513049549874747e-07])
    wrist_2_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_2_rpy), wrist_2_trans)

    wrist_3_trans = np.array(
        [9.062061300900664e-06, 0.09947787349620175, 0.0001411778743239612]
    )
    wrist_3_rpy = np.array([1.572215514545703, 3.141592653589793, 3.141592633687631])
    wrist_3_se3 = pin.SE3(pin.rpy.rpyToMatrix(wrist_3_rpy), wrist_3_trans)

    robot.model.jointPlacements[1] = shoulder_se3
    robot.model.jointPlacements[2] = upper_arm_se3
    robot.model.jointPlacements[3] = forearm_se3
    robot.model.jointPlacements[4] = wrist_1_se3
    robot.model.jointPlacements[5] = wrist_2_se3
    robot.model.jointPlacements[6] = wrist_3_se3
    data = pin.Data(robot.model)
    return robot.model, robot.collision_model, robot.visual_model, data


# this gives me a flying joint for the camera,
# and a million joints for wheels -> it's unusable
# TODO: look what's done in pink, see if this can be usable
# after you've removed camera joint and similar.
def get_heron_model():

    # urdf_path_relative = files('ur_simple_control.robot_descriptions.urdf').joinpath('ur5e_with_robotiq_hande_FIXED_PATHS.urdf')
    urdf_path_absolute = "/home/gospodar/home2/gospodar/lund/praxis/software/ros/ros-containers/home/model.urdf"
    # mesh_dir = files('ur_simple_control')
    # mesh_dir_absolute = os.path.abspath(mesh_dir)
    mesh_dir_absolute = "/home/gospodar/lund/praxis/software/ros/ros-containers/home/heron_description/MIR_robot"

    model = None
    collision_model = None
    visual_model = None
    # this command just calls the ones below it. both are kept here
    # in case pinocchio people decide to change their api.
    # model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path_absolute, mesh_dir_absolute)
    model = pin.buildModelFromUrdf(urdf_path_absolute)
    visual_model = pin.buildGeomFromUrdf(
        model, urdf_path_absolute, pin.GeometryType.VISUAL, None, mesh_dir_absolute
    )
    collision_model = pin.buildGeomFromUrdf(
        model, urdf_path_absolute, pin.GeometryType.COLLISION, None, mesh_dir_absolute
    )

    data = pin.Data(model)

    return model, collision_model, visual_model, data


def get_yumi_model():

    urdf_path_relative = files("ur_simple_control.robot_descriptions").joinpath(
        "yumi.urdf"
    )
    urdf_path_absolute = os.path.abspath(urdf_path_relative)
    # mesh_dir = files('ur_simple_control')
    # mesh_dir_absolute = os.path.abspath(mesh_dir)
    # mesh_dir_absolute = "/home/gospodar/lund/praxis/software/ros/ros-containers/home/heron_description/MIR_robot"
    mesh_dir_absolute = None

    model = None
    collision_model = None
    visual_model = None
    # this command just calls the ones below it. both are kept here
    # in case pinocchio people decide to change their api.
    # model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path_absolute, mesh_dir_absolute)
    model_arms = pin.buildModelFromUrdf(urdf_path_absolute)
    visual_model_arms = pin.buildGeomFromUrdf(
        model_arms, urdf_path_absolute, pin.GeometryType.VISUAL, None, mesh_dir_absolute
    )
    collision_model_arms = pin.buildGeomFromUrdf(
        model_arms,
        urdf_path_absolute,
        pin.GeometryType.COLLISION,
        None,
        mesh_dir_absolute,
    )

    data_arms = pin.Data(model_arms)

    # mobile base as planar joint (there's probably a better
    # option but whatever right now)
    model_mobile_base = pin.Model()
    model_mobile_base.name = "mobile_base"
    geom_model_mobile_base = pin.GeometryModel()
    joint_name = "mobile_base_planar_joint"
    parent_id = 0
    # TEST
    joint_placement = pin.SE3.Identity()
    # joint_placement.rotation = pin.rpy.rpyToMatrix(0, -np.pi/2, 0)
    # joint_placement.translation[2] = 0.2
    MOBILE_BASE_JOINT_ID = model_mobile_base.addJoint(
        parent_id, pin.JointModelPlanar(), joint_placement.copy(), joint_name
    )
    # we should immediately set velocity limits.
    # there are no position limit by default and that is what we want.
    model_mobile_base.velocityLimit[0] = 2
    model_mobile_base.velocityLimit[1] = 2
    model_mobile_base.velocityLimit[2] = 2
    model_mobile_base.effortLimit[0] = 200
    model_mobile_base.effortLimit[1] = 2
    model_mobile_base.effortLimit[2] = 200

    # pretty much random numbers
    # TODO: find heron (mir) numbers
    body_inertia = pin.Inertia.FromBox(30, 0.5, 0.3, 0.4)
    # maybe change placement to sth else depending on where its grasped
    model_mobile_base.appendBodyToJoint(
        MOBILE_BASE_JOINT_ID, body_inertia, pin.SE3.Identity()
    )
    box_shape = fcl.Box(0.5, 0.3, 0.4)
    body_placement = pin.SE3.Identity()
    geometry_mobile_base = pin.GeometryObject(
        "box_shape", MOBILE_BASE_JOINT_ID, box_shape, body_placement.copy()
    )

    geometry_mobile_base.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model_mobile_base.addGeometryObject(geometry_mobile_base)

    # have to add the frame manually
    model_mobile_base.addFrame(
        pin.Frame(
            "mobile_base",
            MOBILE_BASE_JOINT_ID,
            0,
            joint_placement.copy(),
            pin.FrameType.JOINT,
        )
    )

    # frame-index should be 1
    model, visual_model = pin.appendModel(
        model_mobile_base,
        model_arms,
        geom_model_mobile_base,
        visual_model_arms,
        1,
        pin.SE3.Identity(),
    )
    data = model.createData()

    return model, visual_model.copy(), visual_model, data


def heron_approximation():
    # arm + gripper
    model_arm, collision_model_arm, visual_model_arm, data_arm = get_model()

    # mobile base as planar joint (there's probably a better
    # option but whatever right now)
    model_mobile_base = pin.Model()
    model_mobile_base.name = "mobile_base"
    geom_model_mobile_base = pin.GeometryModel()
    joint_name = "mobile_base_planar_joint"
    parent_id = 0
    # TEST
    joint_placement = pin.SE3.Identity()
    
    # joint_placement.rotation = pin.rpy.rpyToMatrix(0, -np.pi/2, 0)
    # joint_placement.translation[2] = 0.2
    # TODO TODO TODO TODO TODO TODO TODO TODO
    # TODO: heron is actually a differential drive,
    # meaning that it is not a planar joint.
    # we could put in a prismatic + revolute joint
    # as the base (both joints being at same position),
    # and that should work for our purposes.
    # this makes sense for initial testing
    # because mobile yumi's base is a planar joint
    MOBILE_BASE_JOINT_ID = model_mobile_base.addJoint(
        parent_id, pin.JointModelPlanar(), joint_placement.copy(), joint_name
    )
    # we should immediately set velocity limits.
    # there are no position limit by default and that is what we want.
    # TODO: put in heron's values
    # TODO: make these parameters the same as in mpc_params in the planner
    model_mobile_base.velocityLimit[0] = 2
    # TODO: PUT THE CONSTRAINTS BACK!!!!!!!!!!!!!!!
    model_mobile_base.velocityLimit[1] = 0
    # model_mobile_base.velocityLimit[1] = 2
    model_mobile_base.velocityLimit[2] = 2
    # TODO: i have literally no idea what reasonable numbers are here
    model_mobile_base.effortLimit[0] = 200
    # TODO: PUT THE CONSTRAINTS BACK!!!!!!!!!!!!!!!
    model_mobile_base.effortLimit[1] = 0
    # model_mobile_base.effortLimit[1] = 2
    model_mobile_base.effortLimit[2] = 200
    # print("OBJECT_JOINT_ID",OBJECT_JOINT_ID)
    # body_inertia = pin.Inertia.FromBox(args.box_mass, box_dimensions[0],
    #        box_dimensions[1], box_dimensions[2])

    # pretty much random numbers
    # TODO: find heron (mir) numbers
    body_placement = pin.SE3.Identity()
    # body_placement.translation[2] -= 0.2
    # body_placement.translation[0] += 0.1
    body_inertia = pin.Inertia.FromBox(30, 0.5, 0.3, 0.4)
    # maybe change placement to sth else depending on where its grasped
    arm2mir = pin.SE3.Identity()
    arm2mir.translation = np.array([-0.061854, -0.0045, 0.872])
    arm2mir.rotation = np.array([0, -1, 0], [1, 0, 0], [0 ,0 ,1])
    model_mobile_base.appendBodyToJoint(
        MOBILE_BASE_JOINT_ID, body_inertia, arm2mir.copy()
    )
    box_shape = fcl.Box(0.5, 0.3, 0.4)
    geometry_mobile_base = pin.GeometryObject(
        "box_shape", MOBILE_BASE_JOINT_ID, box_shape, joint_placement.copy()
    )

    geometry_mobile_base.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model_mobile_base.addGeometryObject(geometry_mobile_base)
    
    # joint_placement.translation[0] = -0.1
    # joint_placement.translation[2] = 0.2
    # have to add the frame manually
    model_mobile_base.addFrame(
        pin.Frame(
            "mobile_base",
            MOBILE_BASE_JOINT_ID,
            0,
            joint_placement.copy(),
            pin.FrameType.JOINT,
        )
    )

    # frame-index should be 1
    model, visual_model = pin.appendModel(
        model_mobile_base,
        model_arm,
        geom_model_mobile_base,
        visual_model_arm,
        1,
        pin.SE3.Identity(),
    )
    data = model.createData()

    # fix gripper
    for geom in visual_model.geometryObjects:
        if "hand" in geom.name:
            s = geom.meshScale
            geom.meshcolor = np.array([1.0, 0.1, 0.1, 1.0])
            # this looks exactly correct lmao
            s *= 0.001
            geom.meshScale = s

    return model, visual_model.copy(), visual_model, data


def mir_approximation():
    # mobile base as planar joint (there's probably a better
    # option but whatever right now)
    model_mobile_base = pin.Model()
    model_mobile_base.name = "mobile_base"
    geom_model_mobile_base = pin.GeometryModel()
    joint_name = "mobile_base_planar_joint"
    parent_id = 0
    # TEST
    joint_placement = pin.SE3.Identity()
    # joint_placement.rotation = pin.rpy.rpyToMatrix(0, -np.pi/2, 0)
    # joint_placement.translation[2] = 0.2
    # TODO TODO TODO TODO TODO TODO TODO TODO
    # TODO: heron is actually a differential drive,
    # meaning that it is not a planar joint.
    # we could put in a prismatic + revolute joint
    # as the base (both joints being at same position),
    # and that should work for our purposes.
    # this makes sense for initial testing
    # because mobile yumi's base is a planar joint
    MOBILE_BASE_JOINT_ID = model_mobile_base.addJoint(
        parent_id, pin.JointModelPlanar(), joint_placement.copy(), joint_name
    )
    # we should immediately set velocity limits.
    # there are no position limit by default and that is what we want.
    # TODO: put in heron's values
    # TODO: make these parameters the same as in mpc_params in the planner
    model_mobile_base.velocityLimit[0] = 2
    model_mobile_base.velocityLimit[1] = 0
    model_mobile_base.velocityLimit[2] = 2
    # TODO: i have literally no idea what reasonable numbers are here
    model_mobile_base.effortLimit[0] = 200
    model_mobile_base.effortLimit[1] = 0
    model_mobile_base.effortLimit[2] = 200
    # print("OBJECT_JOINT_ID",OBJECT_JOINT_ID)
    # body_inertia = pin.Inertia.FromBox(args.box_mass, box_dimensions[0],
    #        box_dimensions[1], box_dimensions[2])

    # pretty much random numbers
    # TODO: find heron (mir) numbers
    body_inertia = pin.Inertia.FromBox(30, 0.5, 0.3, 0.4)
    # maybe change placement to sth else depending on where its grasped
    model_mobile_base.appendBodyToJoint(
        MOBILE_BASE_JOINT_ID, body_inertia, pin.SE3.Identity()
    )
    box_shape = fcl.Box(0.5, 0.3, 0.4)
    body_placement = pin.SE3.Identity()
    geometry_mobile_base = pin.GeometryObject(
        "box_shape", MOBILE_BASE_JOINT_ID, box_shape, body_placement.copy()
    )

    geometry_mobile_base.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model_mobile_base.addGeometryObject(geometry_mobile_base)

    # have to add the frame manually
    # it's tool0 because that's used everywhere
    model_mobile_base.addFrame(
        pin.Frame(
            "tool0",
            MOBILE_BASE_JOINT_ID,
            0,
            joint_placement.copy(),
            pin.FrameType.JOINT,
        )
    )

    data = model_mobile_base.createData()

    return (
        model_mobile_base,
        geom_model_mobile_base.copy(),
        geom_model_mobile_base.copy(),
        data,
    )
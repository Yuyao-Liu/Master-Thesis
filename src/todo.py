import pinocchio as pin
import numpy as np
import hppfcl
from pinocchio.utils import rotate


model = pin.Model()
collision_model = pin.GeometryModel()
visual_model = pin.GeometryModel()

# Add a floating joint
# joint_id = model.addJoint(
#     0,  # Parent joint number, 0 is universe
#     pin.JointModelFreeFlyer(),  # Joint type
#     pin.SE3.Identity(),  # Placement relative to parent joint frame
#     'main_joint'  # Name
#     max_effort=1000 * np.ones(6),  # Limites
#     max_velocity=1000 * np.ones(6),
#     min_config=np.array([-1, -1, -1, 0., 0., 0., 1.]),
#     max_config=np.array([1, 1, 1, 0., 0., 0., 1.]),
# )
joint_id = model.addJoint(
    0,
    pin.JointModelFreeFlyer(),
    pin.SE3.Identity(),
    joint_name='first_joint',
    max_effort=1000 * np.ones(6),
    max_velocity=1000 * np.ones(6),
    min_config=np.array([-1, -1, -1, 0., 0., 0., 1.]),
    max_config=np.array([1, 1, 1, 0., 0., 0., 1.]),
)

# We attach a cylinder to it, in the referential, the base is in 0 and it is along x axis
M_cyl = 3.
h = 0.2
r = 0.02
com = np.array([h/2, 0, 0])  # Where com will be place in joint frame
moment_inertia = np.diag([
    1/2*M_cyl*r**2,
    1/12*M_cyl*h**2 + 1/4*M_cyl*r**2,
    1/12*M_cyl*h**2 + 1/4*M_cyl*r**2
])  # moment inertia matrix of a cylinder

# Add the body as dynamic quantity in the model
# model.appendBodyToJoint(
#     joint_id,  # Joint Id
#     pin.Inertia(M_cyl, com, moment_inertia),  # Inertia matrix with mass, com, moments in express in body frame
#     pin.SE3.Identity()  # transformation from joint frame to body frame
# )
model.appendBodyToJoint(
    joint_id,
    pin.Inertia(M_cyl, com, moment_inertia),
    pin.SE3.Identity()
)

# Add the body as geometric quantity in the collision_model
# geom = pin.GeometryObject(
#     'main_colision_shape',  # Name
#     joint_id,  # joint id
#     hppfcl.Cylinder(r, h),  # Hpp shape
#     pin.SE3(pin.SE3(rotate('y',np.pi/2), np.array([h/2,0,0])))  # Position of mesh in joint frame, here canonically cylinder is along z, we rotate.
# )
geom_col = pin.GeometryObject(
    'world/first_col_shape',
    joint_id,
    hppfcl.Cylinder(r, h),
    pin.SE3(pin.SE3(rotate('y',np.pi/2), np.array([h/2,0,0])))
)
collision_model.addGeometryObject(geom_col)

# But visually it looks like two cylinder with a ball inside
geom_viz1 = pin.GeometryObject(
    'world/first_viz_shape_p1',
    joint_id,
    hppfcl.Cylinder(r, h/2 - r),
    pin.SE3(rotate('y',np.pi/2), np.array([(h/2 - r)/2,0,0]))
)
geom_viz1.meshColor = np.array([1., 0., 0., 1.])
geom_viz2 = pin.GeometryObject(
    'world/first_viz_shape_p2',
    joint_id,
    hppfcl.Cylinder(r, h/2 - r),
    pin.SE3(rotate('y',np.pi/2), np.array([(3*h/2 + r)/2,0,0]))
)
geom_viz2.meshColor = np.array([1., 0., 0., 1.])
geom_viz3 = pin.GeometryObject(
    'world/first_viz_shape_p3',
    joint_id,
    hppfcl.Sphere(r),
    pin.SE3(np.eye(3), np.array([h/2,0,0]))
)
geom_viz3.meshColor = np.array([0., 1., 0., 1.])
visual_model.addGeometryObject(geom_viz1)
visual_model.addGeometryObject(geom_viz2)
visual_model.addGeometryObject(geom_viz3)

# Now let us add another joint at the end of the cylinder
joint_id_2 = model.addJoint(
    joint_id,
    pin.JointModelRY(),
    pin.SE3(np.eye(3), np.array([h,0,0])),
    'second_joint',
    max_effort=np.array([1000]),
    max_velocity=np.array([1000]),
    min_config=np.array([-np.pi]),
    max_config=np.array([np.pi]),

)
model.appendBodyToJoint(
    joint_id_2,
    pin.Inertia(M_cyl, com, moment_inertia),
    pin.SE3.Identity()
)
# Here visual and collision coincide
geom_colviz_2 = pin.GeometryObject(
    'world/second_colviz_shape',
    joint_id_2,
    hppfcl.Cylinder(r, h),
    pin.SE3(pin.SE3(rotate('y',np.pi/2), np.array([h/2,0,0])))
)
collision_model.addGeometryObject(geom_colviz_2)
geom_colviz_2.meshColor = np.array([1., 0., 0., 1.])
visual_model.addGeometryObject(geom_colviz_2)

# And a third one
joint_id_3 = model.addJoint(
    joint_id_2,
    pin.JointModelRY(),
    pin.SE3(np.eye(3), np.array([h,0,0])),
    'third_joint',
    max_effort=np.array([1000]),
    max_velocity=np.array([1000]),
    min_config=np.array([-np.pi]),
    max_config=np.array([np.pi]),
)
model.appendBodyToJoint(
    joint_id_3,
    pin.Inertia(M_cyl, com, moment_inertia),
    pin.SE3.Identity()
)
# Here visual and collision coincide
geom_colviz_3 = pin.GeometryObject(
    'world/third_colviz_shape',
    joint_id_3,
    hppfcl.Cylinder(r, h),
    pin.SE3(pin.SE3(rotate('y',np.pi/2), np.array([h/2,0,0])))
)
collision_model.addGeometryObject(geom_colviz_3)
geom_colviz_3.meshColor = np.array([1., 0., 0., 1.])
visual_model.addGeometryObject(geom_colviz_3)

# But we can have object that are not in chain, tree is alowed !
joint_id_bis = model.addJoint(
    0,
    pin.JointModelFreeFlyer(),
    pin.SE3(np.eye(3), np.array([0,0.1,0])),
    'other_joint',
    max_effort=1000 * np.ones(6),
    max_velocity=1000 * np.ones(6),
    min_config=np.array([-1, -1, -1, 0., 0., 0., 1.]),
    max_config=np.array([1, 1, 1, 0., 0., 0., 1.]),
)

# We attach a ball to it, in the referential, the base is in 0 and it is along x axis
M_ball = 3.
r_ball = 0.05
com_ball = np.array([0, 0, 0])  # Where com will be place in joint frame
moment_inertia_ball = M_ball * np.eye(3)  # moment inertia matrix of a cylinder

model.appendBodyToJoint(
    joint_id_bis,
    pin.Inertia(M_ball, com_ball, moment_inertia_ball),
    pin.SE3.Identity()
)

geom_col_other = pin.GeometryObject(
    'world/other_colviz_shape',
    joint_id_bis,
    hppfcl.Sphere(2*r),
    pin.SE3(pin.SE3(rotate('y',np.pi/2), np.array([h/2,0,0])))
)
collision_model.addGeometryObject(geom_col_other)
geom_colviz_3.meshColor = np.array([0., 1., 1., 1.])
visual_model.addGeometryObject(geom_col_other)
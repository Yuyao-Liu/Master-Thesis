from setuptools import setup, find_packages

setup(
    name="smc",
    version="0.3",
    description="Simple Manipulator Control (SMC) - simplest possible framework for robot control.",
    author="Marko Guberina",
    url="https://gitlab.control.lth.se/marko-g/ur_simple_control",
    packages=["smc"],
    #        package_dir={"": "ur_simple_control"},
    package_data={
        "smc.robots.robot_descriptions": ["*"],
    },
    zip_safe=False,
)
# NOTE: if you want to switch to the toml thing,
# here you go, knock yourself out, didn't really work for me,
# and i don't care at this stage
# add other ones and test, check .toml file for more
# dependencies need to be taken care of separately
# because you don't want to install pinocchio via pip
# install_requires=['numpy'],
# broken, but if you'll want to switch later here you go
# packages=find_packages(where='src'),
# package_data={'ur_simple_control': ['clik/*', 'dmp/*', 'util/*']}
# )

# dependencies:
# pinocchio - robot math
# numpy - math
# matplotlib - plotter
# meshcat, meshcat_shapes - visualizer
# [optional] UR robots control: ur_rtde
# [optional] ik solvers: qpsolvers, quadprog, proxsuite, ecos, pink
# [optional] for opc/mpc: crocoddyl
# [optional] for path planning: albin's path planning repos: starworlds, star_navigation, tunnel_mpc (NOTE all these repos are copy-pasted to path_generation atm)
# [optional] for visual servoing: opencv-python
# [optional] for different ocp: casadi (requires pinocchio > 3.0)
# [optional] for yumi, mobile yumi, and heron: ros2

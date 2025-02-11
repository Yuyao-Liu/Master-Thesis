from setuptools import setup, find_packages
setup(name='ur_simple_control',
        version='0.1',
        description='Collection of control algorithms for the UR5e arm based on the ur_rtde interface for communication and pinocchio for calculations.',
        author='Marko Guberina',
        url='https://gitlab.control.lth.se/marko-g/ur_simple_control',
        packages=['ur_simple_control'],
        #        package_dir={"": "ur_simple_control"},
        package_data={
            'ur_simple_control.robot_descriptions': ['*'],
            },
        zip_safe=False)
# NOTE: if you want to switch to the toml thing,
# here you go, knock yourself out, didn't really work for me,
# and i don't care at this stage
        # add other ones and test, check .toml file for more
        # dependencies need to be taken care of separately
        # because you don't want to install pinocchio via pip
        #install_requires=['numpy'],
        # broken, but if you'll want to switch later here you go
        #packages=find_packages(where='src'),
        #package_data={'ur_simple_control': ['clik/*', 'dmp/*', 'util/*']}
        #)

# dependencies:
# numpy 
# matplotlib
# qpsolvers
# quadprog
# pinocchio
# meshcat
# meshcat_shapes
# crocoddyl 
# albin's path planning repos: starworlds, star_navigation, tunnel_mpc
# optional for camera: cv2
# optional for different ocp: casadi AND pinocchio > 3.0

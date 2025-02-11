#!/bin/bash
# the idea here is to run all the runnable things
# and test for syntax errors 

############
#  camera  #
############
# the only one where you check plotter
runnable="camera_no_lag.py --max-iterations=1500 --no-visualize-manipulator"
echo $runnable
python $runnable
echo "=========================================================="

###################
#  classic cliks  #
###################
# the only one where you check visualizer
# damped pseudoinverse arm
runnable="clik.py --randomly-generate-goal"
echo $runnable
python $runnable

# damped pseudoinverse whole body mobile
runnable="clik.py --robot=heron --randomly-generate-goal --fast-simulation --no-visualize-manipulator --no-real-time-plotting --max-iterations=2000"
echo $runnable
python $runnable

# QP arm
runnable="clik.py --randomly-generate-goal --clik-controller=invKinmQP --fast-simulation --no-visualize-manipulator --no-real-time-plotting --max-iterations=2000"
echo $runnable
python $runnable

# QP whole body mobile
runnable="clik.py --robot=heron --randomly-generate-goal --clik-controller=invKinmQP --fast-simulation --no-visualize-manipulator --no-real-time-plotting --max-iterations=2000"
echo $runnable
python $runnable

# cart pulling mpc
runnable="cart_pulling.py --max-solver-iter=10 --n-knots=30 --robot=heron --past-window-size=200"
echo $runnable
python $runnable


#python cart_pulling.py
#python casadi_ocp_collision_avoidance.py
#python challenge_main.py
#python comparing_logs_example.py
#python crocoddyl_mpc.py
#python crocoddyl_ocp_clik.py
#python data
#python drawing_from_input_drawing.py
#python force_control_test.py
#python graz
#python heron_pls.py
#python joint_trajectory.csv
#python log_analysis_example.py
#python old_or_experimental
#python path_following_mpc.py
#python path_in_pixels.csv
#python pin_contact3d.py
#python plane_pose.pickle
#python plane_pose.pickle_save
#python plane_pose_sim.pickle
#python point_force_control.py
#python point_impedance_control.py
#python pushing_via_friction_cones.py
#python __pycache__
#python ros2_clik.py
#python smc_node.py
#python test_by_running.sh
#python test_crocoddyl_opt_ctrl.py
#python wiping_path.csv_save
#

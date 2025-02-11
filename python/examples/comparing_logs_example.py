from ur_simple_control.visualize.manipulator_comparison_visualizer import getLogComparisonArgs, ManipulatorComparisonManager
import time

print("""
this code animates 2 manipulator and can run an animated
plot along side it with the same timing.
you're supposed to select what you want to plot yourself.
the comparison manager has no idea how many control loops
you have nor what's logged in them, apart from the fact 
that everything has to have qs to visualize the manipulators.
here we're assuming you have 1 control loop per log,
and that the same things are logged.
also, you need to provide the two log files
with --log-file1=/path/to/file1 and
--log-file1=/path/to/file2
""")

args = getLogComparisonArgs()
cmp_manager = ManipulatorComparisonManager(args)

key = list(cmp_manager.logm1.loop_logs.keys())[0]
#cmp_manager.createRunningPlot(cmp_manager.logm1.loop_logs[key], 0, len(cmp_manager.logm1.loop_logs[key]['qs']))
#cmp_manager.createRunningPlot(cmp_manager.logm2.loop_logs[key], 0, len(cmp_manager.logm2.loop_logs[key]['qs']))
cmp_manager.visualizeWholeRuns()
cmp_manager.visualizer_manager.sendCommand("befree")
time.sleep(100)
print("main done")
time.sleep(0.1)
cmp_manager.visualizer_manager.terminateProcess()
if args.debug_prints:
    print("terminated manipulator_visualizer_process")

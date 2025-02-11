# this is just random code which needs to be put into 
# a concrete example.
# but the chunks and some comments make sense,
# so they're kept here for future use

#######################################################################
#                 some force_mode from ur api chucks                  #
#######################################################################

# ========================================================================
# TODO: either write a separate file where this is used
# ---> TODO write a separate file where you're testing this out
# or just delete it
task_frame = [0, 0, 0, 0, 0, 0]
# these are in {0,1} and select which task frame direction compliance is active in
# just be soft everywhere
selection_vector = [1, 1, 1, 1, 1, 1]
# the wrench applied to the environment: 
# position is adjusted to achieve the specified wrench
# let's pretend speedjs are this and see what happens (idk honestly)
wrench = [0, 0, 0, 0, 0, 0]
ftype = 2
# limits for:
# - compliant axes: highest tcp velocities allowed on compliant axes
# - non-compliant axes: maximum tcp position error compared to the program (which prg,
#                       and where is this set?)
# why these values?
limits = [2, 2, 2, 2, 2, 2]
wrench_avg = np.zeros((5,6))

# TODO: move to existing force mode API testing file
# this is very stupind, but it was quick to implement
vel_cmd8 = list(vel_cmd)
vel_cmd8.append(0.0)
vel_cmd8.append(0.0)
vel_cmd8 = np.array(vel_cmd8)
vel_tcp = J @ vel_cmd8
vel_tcp = vel_tcp * 10
rtde_control.forceMode(task_frame, selection_vector, vel_tcp, ftype, limits)
# ========================================================================

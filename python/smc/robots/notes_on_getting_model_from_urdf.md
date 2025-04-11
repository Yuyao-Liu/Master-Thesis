"""
possible improvement: 
    get the calibration file of the robot without ros
    and then put it here.
    these magic numbers are not a good look.
"""

# can't get the urdf reading with these functions to save my life, idk what or why

#############################################################
# PACKAGE_DIR IS THE WHOLE smc FOLDER (cos that's accessible from anywhere it's installed)
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

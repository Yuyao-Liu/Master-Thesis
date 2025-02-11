# PYTHON_ARGCOMPLETE_OK
import numpy as np
import pinocchio as pin

def getRandomlyGeneratedGoal(args):
    Mgoal = pin.SE3.Random()
    # has to be close
    translation = np.random.random(3) * 0.4
    translation[2] = np.abs(translation[2])
    translation = translation + np.ones(3) * 0.1
    Mgoal.translation = translation
    if args.debug_prints:
        print(Mgoal)
    return Mgoal

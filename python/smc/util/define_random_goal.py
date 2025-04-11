import numpy as np
from pinocchio import SE3


def getRandomlyGeneratedGoal(args):
    T_w_goal = SE3.Random()
    # has to be close
    translation = np.random.random(3) * 0.8 - 0.4
    translation[2] = np.abs(translation[2])
    translation = translation + np.ones(3) * 0.1
    T_w_goal.translation = translation
    if args.debug_prints:
        print(T_w_goal)
    return T_w_goal

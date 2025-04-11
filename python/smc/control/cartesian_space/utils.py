import argparse


def getClikArgs(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    getClikArgs
    ------------
    Every clik-related magic number, flag and setting is put in here.
    This way the rest of the code is clean.
    Force filtering is considered a core part of these control loops
    because it's not necessarily the same as elsewhere.
    If you want to know what these various arguments do, just grep
    though the code to find them (but replace '-' with '_' in multi-word arguments).
    All the sane defaults are here, and you can change any number of them as an argument
    when running your program. If you want to change a drastic amount of them, and
    not have to run a super long commands, just copy-paste this function and put your
    own parameters as default ones.
    """
    # parser = argparse.ArgumentParser(description='Run closed loop inverse kinematics \
    #        of various kinds. Make sure you know what the goal is before you run!',
    #        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ####################
    #  clik arguments  #
    ####################
    parser.add_argument(
        "--goal-error",
        type=float,
        help="the final position error you are happy with",
        default=1e-2,
    )
    parser.add_argument(
        "--tikhonov-damp",
        type=float,
        help="damping scalar in tikhonov regularization",
        default=1e-3,
    )
    # TODO add the rest
    parser.add_argument(
        "--ik-solver",
        type=str,
        help="select which click algorithm you want",
        default="dampedPseudoinverse",
        choices=[
            "dampedPseudoinverse",
            "jacobianTranspose",
            "QPquadprog",
            "QPproxsuite",
            "QPManipMax",
        ],
    )

    ###########################################
    #  force sensing and feedback parameters  #
    ###########################################
    parser.add_argument(
        "--alpha",
        type=float,
        help="force feedback proportional coefficient",
        default=0.01,
    )
    parser.add_argument(
        "--beta", type=float, help="low-pass filter beta parameter", default=0.01
    )
    parser.add_argument(
        "--kp",
        type=float,
        help="proportial control constant for position errors",
        default=1.0,
    )
    # NOTE: unnecessary on velocity controlled robots
    parser.add_argument(
        "--kv", type=float, help="damping in impedance control", default=0.001
    )
    parser.add_argument(
        "--z-only",
        action=argparse.BooleanOptionalAction,
        help="whether you have general impedance or just ee z axis",
        default=False,
    )

    ###############################
    #  path following parameters  #
    ###############################
    parser.add_argument(
        "--max-init-clik-iterations",
        type=int,
        help="number of max clik iterations to get to the first point",
        default=10000,
    )
    parser.add_argument(
        "--max-running-clik-iterations",
        type=int,
        help="number of max clik iterations between path points",
        default=1000,
    )
    parser.add_argument(
        "--viz-test-path",
        action=argparse.BooleanOptionalAction,
        help="number of max clik iterations between path points",
        default=False,
    )

    return parser

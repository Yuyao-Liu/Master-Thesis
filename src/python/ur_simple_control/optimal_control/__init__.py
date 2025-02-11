import importlib.util
#if importlib.util.find_spec('casadi'):
#    import pinocchio as pin
#    if int(pin.__version__[0]) < 3:
#        print("you need to have pinocchio version 3.0.0 or greater to use pinocchio.casadi!")
#        exit()
#    from .create_pinocchio_casadi_ocp import *
from .crocoddyl_mpc import *
from .crocoddyl_optimal_control import *
from .get_ocp_args import *

import importlib.util
if importlib.util.find_spec('star_navigation') and \
        importlib.util.find_spec('starworld_tunnel_mpc') and \
        importlib.util.find_spec('starworlds'):
    from .planner import *
#import starworlds

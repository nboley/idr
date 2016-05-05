import sys

__version__ = "2.0.3"

DEBUG_LEVELS = {'ERROR', 'WARNING', None, 'VERBOSE', 'DEBUG'}
ERROR_LEVELS = {'ERROR', 'WARNING'}

log_ofp = sys.stderr
def log(*args, level=None):
    assert level in DEBUG_LEVELS
    args = [str(x) for x in args]
    if args[-1] in DEBUG_LEVELS: 
        if level == None:
            level = args[-1]
        args = args[:-1]
    if level in ERROR_LEVELS:
        print(" ".join(args), file=sys.stderr)
        sys.stderr.flush()
        if log_ofp == sys.stderr: return
    elif QUIET: 
        return
    
    if (level in ('ERROR', 'WARNING', None) 
            or (level == 'VERBOSE' and VERBOSE)):
        print(" ".join(args), file=log_ofp)
        log_ofp.flush()

## Global config options
VERBOSE = False
QUIET = False
PROFILE = False

## idr.py config options
MAX_ITER_DEFAULT = 3000
CONVERGENCE_EPS_DEFAULT = 1e-6

DEFAULT_MU = 0.1
DEFAULT_SIGMA = 1.0
DEFAULT_RHO = 0.2
DEFAULT_MIX_PARAM = 0.5

DEFAULT_SOFT_IDR_THRESH = 0.05
DEFAULT_IDR_THRESH = 1.00

FILTER_PEAKS_BELOW_NOISE_MEAN = True
ONLY_ALLOW_NON_NEGATIVE_VALUES = True

## optimization.py config options

MIN_MIX_PARAM = 0.01
MAX_MIX_PARAM = 0.99

MIN_RHO = 0.10
MAX_RHO = 0.99

MIN_SIGMA = 0.20
MAX_SIGMA = 20

MIN_MU = 0.0
MAX_MU = 20.0

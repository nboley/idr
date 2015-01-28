Peak = namedtuple('Peak', ['chrm', 'strand', 'start', 'stop', 'signal'])

def log(msg, level=None):
    if QUIET: return
    if level == None or (level == 'VERBOSE' and VERBOSE):
        print(msg, file=sys.stderr)

## Global config options
VERBOSE = False
QUIET = False
PROFILE = False


## idr.py config options
IGNORE_NONOVERLAPPING_PEAKS = False

MAX_ITER_DEFAULT = 10000
CONVERGENCE_EPS_DEFAULT = 1e-6

DEFAULT_MU = 0.1
DEFAULT_SIGMA = 1.0
DEFAULT_RHO = 0.2
DEFAULT_MIX_PARAM = 0.5

FILTER_PEAKS_BELOW_NOISE_MEAN = True

## optimization.py config options

MIN_MIX_PARAM = 0.01
MAX_MIX_PARAM = 0.99

MIN_RHO = 0.20
MAX_RHO = 0.99

MIN_SIGMA = 0.20

MIN_MU = 0.0

import os, sys

import math

import numpy

from scipy.stats.stats import rankdata

def mean(items):
    items = list(items)
    return sum(items)/float(len(items))

from collections import namedtuple, defaultdict, OrderedDict
from itertools import chain
Peak = namedtuple('Peak', ['chrm', 'strand', 'start', 'stop', 'signal'])

VERBOSE = False
QUIET = False
PROFILE = False

IGNORE_NONOVERLAPPING_PEAKS = False

MAX_ITER_DEFAULT = 1000
CONVERGENCE_EPS_DEFAULT = 1e-6

DEFAULT_MU = 2.0
DEFAULT_SIGMA = 0.8
DEFAULT_RHO = 0.6
DEFAULT_MIX_PARAM = 0.5

import idr.optimization
from idr.optimization import estimate_model_params, old_estimator
from idr.utility import calc_post_membership_prbs, compute_pseudo_values


def load_bed(fp, signal_type):
    signal_index = {"signal.value": 6, "p.value": 7, "q.value": 8}[signal_type]
    grpd_peaks = defaultdict(list)
    for line in fp:
        if line.startswith("#"): continue
        if line.startswith("track"): continue
        data = line.split()
        signal = float(data[signal_index])
        if signal < 0: 
            raise ValueError("Invalid {}: {:e}".format(signal_type, signal))
        peak = Peak(data[0], data[5], int(data[1]), int(data[2]), signal )
        grpd_peaks[(peak.chrm, peak.strand)].append(peak)
    return grpd_peaks

def merge_peaks_in_contig(s1_peaks, s2_peaks, pk_agg_fn):
    """Merge peaks in a single contig/strand.
    
    returns: The merged peaks. 
    """
    # merge and sort all peaks, keeping track of which sample they originated in
    all_intervals = sorted(chain(
            ((pk.start,pk.stop,pk.signal,1) for i, pk in enumerate(s1_peaks)),
            ((pk.start,pk.stop,pk.signal,2) for i, pk in enumerate(s2_peaks))))
    
    # grp overlapping intervals. Since they're already sorted, all we need
    # to do is check if the current interval overlaps the previous interval
    grpd_intervals = [[],]
    curr_start, curr_stop = all_intervals[0][:2]
    for x in all_intervals:
        if x[0] < curr_stop:
            curr_stop = max(x[1], curr_stop)
            grpd_intervals[-1].append(x)
        else:
            curr_start, curr_stop = x[:2]
            grpd_intervals.append([x,])

    # build the unified peak list, setting the score to 
    # zero if it doesn't exist in both replicates
    merged_pks = []
    for intervals in grpd_intervals:
        # grp peaks by their source, and calculate the merged
        # peak boundaries
        grpd_peaks = OrderedDict(((1, []), (2, [])))
        pk_start, pk_stop = 1e9, -1
        for x in intervals:
            pk_start = min(x[0], pk_start)
            pk_stop = max(x[0], pk_stop)
            grpd_peaks[x[3]].append(x)

        # skip regions that dont have a peak in all replicates
        if IGNORE_NONOVERLAPPING_PEAKS:
            if any(0 == len(peaks) for peaks in grpd_peaks.values()):
                continue

        s1, s2 = (pk_agg_fn(pk[2] for pk in pks) for pks in grpd_peaks.values())
        merged_pk = (pk_start, pk_stop, s1, s2, grpd_peaks)
        merged_pks.append(merged_pk)
    
    return merged_pks

def merge_peaks(s1_peaks, s2_peaks, pk_agg_fn):
    """Merge peaks over all contig/strands
    
    """
    contigs = sorted(set(chain(s1_peaks.keys(), s2_peaks.keys())))
    merged_peaks = []
    for key in contigs:
        # since s*_peaks are default dicts, it will never raise a key error, 
        # but instead return an empty list which is what we want
        merged_peaks.extend(
            key + pk for pk in merge_peaks_in_contig(
                s1_peaks[key], s2_peaks[key], pk_agg_fn))
    
    merged_peaks.sort(key=lambda x:pk_agg_fn((x[4],x[5])), reverse=True)
    return merged_peaks

def build_rank_vectors(merged_peaks):
    # allocate memory for the ranks vector
    s1 = numpy.zeros(len(merged_peaks))
    s2 = numpy.zeros(len(merged_peaks))
    # add the signal
    for i, x in enumerate(merged_peaks):
        s1[i], s2[i] = x[4], x[5]

    rank1 = numpy.lexsort((numpy.random.random(len(s1)), s1)).argsort()
    rank2 = numpy.lexsort((numpy.random.random(len(s2)), s2)).argsort()
    
    return ( numpy.array(rank1, dtype=numpy.int), 
             numpy.array(rank2, dtype=numpy.int) )

def build_idr_output_line(
    contig, strand, signals, merged_peak, IDR, localIDR):
    rv = [contig,]
    for signal, key in zip(signals, (1,2)):
        if len(merged_peak[key]) == 0: 
            rv.extend(("-1", "-1"))
        else:
            rv.append( "%i" % min(x[0] for x in merged_peak[key]))
            rv.append( "%i" % max(x[1] for x in merged_peak[key]))
        rv.append( "%.5f" % signal )
    
    rv.append("%.5f" % IDR)
    rv.append("%.5f" % localIDR)
    rv.append(strand)
        
    return "\t".join(rv)

def calc_IDR(theta, r1, r2):
    """
    idr <- 1 - e.z
    o <- order(idr)
    idr.o <- idr[o]
    idr.rank <- rank(idr.o, ties.method = "max")
    top.mean <- function(index, x) {
        mean(x[1:index])
    }
    IDR.o <- sapply(idr.rank, top.mean, idr.o)
    IDR <- idr
    IDR[o] <- IDR.o
    """
    mu, sigma, rho, p = theta
    z1 = compute_pseudo_values(r1, mu, sigma, p)
    z2 = compute_pseudo_values(r2, mu, sigma, p)
    localIDR = 1 - calc_post_membership_prbs(numpy.array(theta), z1, z2)
    local_idr_order = localIDR.argsort()
    ordered_local_idr = localIDR[local_idr_order]
    ordered_local_idr_ranks = rankdata( ordered_local_idr, method='max' )
    IDR = []
    for rank in ordered_local_idr_ranks:
        IDR.append(ordered_local_idr[:rank].mean())
    IDR = numpy.array(IDR)

    return localIDR, IDR[local_idr_order]

def fit_model_and_calc_idr(r1, r2, 
                           starting_point=None,
                           max_iter=MAX_ITER_DEFAULT, 
                           convergence_eps=CONVERGENCE_EPS_DEFAULT, 
                           fix_mu=False, fix_sigma=False ):
    # in theory we would try to find good starting point here,
    # but for now just set it to somethign reasonable
    if starting_point == None:
        starting_point = (DEFAULT_MU, DEFAULT_SIGMA, 
                          DEFAULT_RHO, DEFAULT_MIX_PARAM)
    
    log("Initial parameter values: [%s]" % " ".join(
        "%.2f" % x for x in starting_point))
    
    # fit the model parameters    
    log("Fitting the model parameters", 'VERBOSE');
    if PROFILE:
            import cProfile
            cProfile.runctx("""theta, loss = estimate_model_params(
                                    r1,r2,
                                    starting_point, 
                                    max_iter=max_iter, 
                                    convergence_eps=convergence_eps,
                                    fix_mu=fix_mu, fix_sigma=fix_sigma)
                                   """, 
                            {'estimate_model_params': estimate_model_params}, 
                            {'r1':r1, 'r2':r2, 
                             'starting_point': starting_point,
                             'max_iter': max_iter, 
                             'convergence_eps': convergence_eps,
                             'fix_mu': fix_mu, 'fix_sigma': fix_sigma} )
            assert False
    theta, loss = estimate_model_params(
        r1, r2,
        starting_point, 
        max_iter=max_iter, 
        convergence_eps=convergence_eps,
        fix_mu=fix_mu, fix_sigma=fix_sigma)
    
    log("Finished running IDR on the datasets", 'VERBOSE')
    log("Final parameter values: %s" % " ".join("%.2f" % x for x in theta))
    
    # calculate the global IDR
    localIDRs, IDRs = calc_IDR(numpy.array(theta), r1, r2)

    return localIDRs, IDRs

def write_results_to_file(merged_peaks, output_file, 
                          max_allowed_idr=1.0,
                          localIDRs=None, IDRs=None):
    # write out the result
    log("Writing results to file", "VERBOSE");
    
    if localIDRs == None or IDRs == None:
        assert IDRs == None
        assert localIDRs == None
        localIDRs = numpy.ones(len(merged_peaks))
        IDRs = numpy.ones(len(merged_peaks))

    
    num_peaks_passing_thresh = 0
    for localIDR, IDR, merged_peak in zip(
            localIDRs, IDRs, merged_peaks):
        # skip peaks with global idr values below the threshold
        if IDR > max_allowed_idr: continue
        num_peaks_passing_thresh += 1
        opline = build_idr_output_line(
            merged_peak[0], merged_peak[1], 
            merged_peak[4:6], 
            merged_peak[6], IDR, localIDR )
        print( opline, file=output_file )

    log("Number of peaks passing IDR cutoff of {} - {}/{} ({:.1f}%)\n".format(
        max_allowed_idr, 
        num_peaks_passing_thresh, len(merged_peaks),
        100*float(num_peaks_passing_thresh)/len(merged_peaks))
    )
    
    return 

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="""
Program: IDR (Irreproducible Discovery Rate)
Version: {PACKAGE_VERSION}\n
Contact: Nikhil R Podduturi <nikhilrp@stanford.edu>
         Nathan Boley <npboley@gmail.com>

""")

    parser.add_argument( '-a', type=argparse.FileType("r"), required=True,
        help='narrowPeak or broadPeak file containing peaks from sample 1.')

    parser.add_argument( '-b', type=argparse.FileType("r"), required=True,
        help='narrowPeak or broadPeak file containing peaks from sample 2.')

    default_ofname = "idrValues.txt"
    parser.add_argument( '--output-file', "-o", type=argparse.FileType("w"), 
                         default=open(default_ofname, "w"), 
        help='File to write output to. default: {}'.format(default_ofname))

    parser.add_argument( '--idr', "-i", type=float, default=1.0, 
        help='Only report peaks with a global idr threshold below this value. Default: report all peaks')

    parser.add_argument( '--rank', default="signal.value",
                         choices=["signal.value", "p.value", "q.value"],
                         help='Type of ranking measure to use.')
    
    parser.add_argument( '--use-nonoverlapping-peaks', 
                         action="store_true", default=False,
        help='Use peaks without an overlapping match and set the value to 0.')
    
    parser.add_argument( '--peak-merge-method', 
                         choices=["sum", "avg", "min", "max"], default=None,
        help="Which method to use for merging peaks.\n" \
              + "\tDefault: 'mean' for signal, 'min' for p/q-value.")

    parser.add_argument( '--fix-mu', action='store_true', 
        help="Fix mu to the starting point and do not let it vary.")    
    parser.add_argument( '--fix-sigma', action='store_true', 
        help="Fix sigma to the starting point and do not let it vary.")    

    parser.add_argument( '--initial-mu', type=float, default=DEFAULT_MU,
        help="Initial value of mu. Default: %.2f" % DEFAULT_MU)
    parser.add_argument( '--initial-sigma', type=float, default=DEFAULT_SIGMA,
        help="Initial value of sigma. Default: %.2f" % DEFAULT_SIGMA)
    parser.add_argument( '--initial-rho', type=float, default=DEFAULT_RHO,
        help="Initial value of rho. Default: %.2f" % DEFAULT_RHO)
    parser.add_argument( '--initial-mix-param', 
        type=float, default=DEFAULT_MIX_PARAM,
        help="Initial value of the mixture params. Default: %.2f" \
                         % DEFAULT_MIX_PARAM)


    parser.add_argument( '--max-iter', type=int, default=MAX_ITER_DEFAULT, 
        help="The maximum number of optimization iterations. Default: %i" 
                         % MAX_ITER_DEFAULT)
    parser.add_argument( '--convergence-eps', type=float, 
                         default=CONVERGENCE_EPS_DEFAULT, 
        help="The maximum change in parameter value changes for convergence. Default: %.2e" 
                         % CONVERGENCE_EPS_DEFAULT)

    parser.add_argument( '--only-merge-peaks', action='store_true', 
        help="Only return the merged peak list.")    


    parser.add_argument( '--verbose', action="store_true", default=False, 
                         help="Print out additional debug information")
    parser.add_argument( '--quiet', action="store_true", default=False, 
                         help="Don't print any status messages")

    args = parser.parse_args()

    global VERBOSE
    if args.verbose: 
        VERBOSE = True 

    global QUIET
    if args.quiet: 
        QUIET = True 
        VERBOSE = False
    
    idr.optimization.VERBOSE = VERBOSE

    global IGNORE_NONOVERLAPPING_PEAKS
    IGNORE_NONOVERLAPPING_PEAKS = not args.use_nonoverlapping_peaks

    # decide what aggregation function to use for peaks that need to be merged
    if args.peak_merge_method == None:
        peak_merge_fn = {"signal.value": mean, "q.value": mean, "p.value": mean}[
            args.rank]
    else:
        peak_merge_fn = {"sum": sum, "avg": mean, "min": min, "max": max}[
            args.peak_merge_method]

    return args, peak_merge_fn

def log(msg, level=None):
    if QUIET: return
    if level == None or (level == 'VERBOSE' and VERBOSE):
        print(msg, file=sys.stderr)

def main():
    args, peak_merge_fn = parse_args()
    
    # load the peak files
    log("Loading the peak files", 'VERBOSE')
    f1 = load_bed(args.a, args.rank)
    f2 = load_bed(args.b, args.rank)

    # build a unified peak set
    log("Merging peaks", 'VERBOSE')
    merged_peaks = merge_peaks(f1, f2, peak_merge_fn)
    
    # build the ranks vector
    log("Ranking peaks", 'VERBOSE')
    r1, r2 = build_rank_vectors(merged_peaks)
    
    if args.only_merge_peaks:
        localIDRs, IDRs = None, None
    else:
        if len(merged_peaks) < 20:
            error_msg = "Peak files must contain at least 20 peaks post-merge"
            error_msg += "\nHint: Merged peaks were written to the output file"
            write_results_to_file(
                merged_peaks, args.output_file )
            raise ValueError(error_msg)

        localIDRs, IDRs = fit_model_and_calc_idr(
            r1, r2, 
            starting_point=(
                args.initial_mu, args.initial_sigma, 
                args.initial_rho, args.initial_mix_param),
            max_iter=args.max_iter,
            convergence_eps=args.convergence_eps,
            fix_mu=args.fix_mu, fix_sigma=args.fix_sigma )    
    
    write_results_to_file(merged_peaks, 
                          args.output_file, 
                          max_allowed_idr=args.idr,
                          localIDRs=localIDRs, IDRs=IDRs)

    args.output_file.close()

if __name__ == '__main__':
    main()

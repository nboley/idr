import os, sys

from scipy.optimize import fminbound

import numpy

import math

VERBOSE = False

## These functions calcualte the the pseudo_lhd and pseudo_gradient,
## that is they calculate the derivative through G^-1
## 
## this doesn't work because the likelihood is not a good measure of
## model fit.
#import symbolic
#(calc_pseudo_log_lhd, calc_pseudo_log_lhd_gradient
# ) = symbolic.build_mixture_loss_and_grad(build_pseudo_functions=True)

## These are symbolic functions to test the precision and accuracy
## of the faster versions included below
#
#import symbolic
#(calc_gaussian_mix_log_lhd, calc_gaussian_mix_log_lhd_gradient
# ) = symbolic.build_mixture_loss_and_grad()

MIN_MIX_PARAM = 0.01
MAX_MIX_PARAM = 0.99

MIN_RHO = 0.20
MAX_RHO = 0.99

MIN_SIGMA = 0.20

MIN_MU = 0.0

from idr.utility import (
    compute_pseudo_values, 
    calc_post_membership_prbs, 
    calc_gaussian_mix_log_lhd )

def log_lhd_loss(r1, r2, theta):
    mu, sigma, rho, p = theta
    z1 = compute_pseudo_values(r1, mu, sigma, p)
    z2 = compute_pseudo_values(r2, mu, sigma, p)
    return -calc_gaussian_mix_log_lhd(theta, z1, z2)


def sum_grad_sq_loss(r1, r2, theta):
    mu, sigma, rho, p = theta
    z1 = compute_pseudo_values(r1, mu, sigma, p)
    z2 = compute_pseudo_values(r2, mu, sigma, p)
    grad = calc_pseudo_log_lhd_gradient(theta, z1, z2, False, False)
    return (grad**2).sum()

# set the loss function to log_lhd
calc_loss = log_lhd_loss


def old_estimator(ranks_1, ranks_2):
    import ctypes
    
    class c_OptimizationRV(ctypes.Structure):
        _fields_ = [("n_iters", ctypes.c_int), 
                    ("rho", ctypes.c_double), 
                    ("p", ctypes.c_double)]

    C_em_gaussian = ctypes.cdll.LoadLibrary(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                     "IDR_parameter_estimation.so")).em_gaussian
    C_em_gaussian.restype = c_OptimizationRV
    
    n = len(ranks_1)
    assert( n == len(ranks_1) == len(ranks_2) )
    localIDR = numpy.zeros(n, dtype='d')
    rv = C_em_gaussian(
        ctypes.c_int(n), 
        ranks_1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ranks_2.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        localIDR.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        (ctypes.c_int(1) if VERBOSE else ctypes.c_int(0)) )

    theta = (0.0, 1.0, float(rv.rho), float(rv.p)) 
    return theta, log_lhd_loss(ranks_1, ranks_2, theta )


def EM_step(z1, z2, starting_point, fix_mu=False, fix_sigma=False):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    ez = calc_post_membership_prbs(starting_point, z1, z2)
    
    # just a small optimization
    ez_sum = ez.sum()
        
    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    mu = (mu_1 + mu_2)/2
    
    weighted_sum_sqs_1 = (ez*((z1-mu)**2)).sum()
    weighted_sum_sqs_2 = (ez*((z2-mu)**2)).sum()
    weighted_sum_prod = (ez*(z2-mu)*(z1-mu)).sum()

    sigma = math.sqrt((weighted_sum_sqs_1+weighted_sum_sqs_2)/(2*ez_sum))
    
    rho = 2*(ez*(z1-mu)*(z2-mu)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)

    p = ez_sum/len(ez)
    
    if fix_mu: mu = i_mu
    if fix_sigma: sigma = i_sigma
    return numpy.array([mu, sigma, rho, p])

def grid_search(r1, r2 ):
    res = []
    best_theta = None
    max_log_lhd = -1e100
    for mu in numpy.linspace(0.1, 5, num=10):
        for sigma in numpy.linspace(0.5, 3, num=10):
            for rho in numpy.linspace(0.1, 0.9, num=10):
                for pi in numpy.linspace(0.1, 0.9, num=10):
                    z1 = compute_pseudo_values(r1, mu, sigma, pi)
                    z2 = compute_pseudo_values(r2, mu, sigma, pi)
                    log_lhd = calc_gaussian_mix_log_lhd((mu, sigma, rho, pi), z1, z2)
                    if log_lhd > max_log_lhd:
                        best_theta = ((mu,mu), (sigma,sigma), rho, pi)
                        max_log_lhd = log_lhd
    
    return best_theta

def find_max_step_size(param_val, grad_val, limit_to_1 = False, MIN_VAL=1e-6):
    if grad_val < 0 and param_val < MIN_VAL: return 0
    if limit_to_1 and grad_val > 0 and param_val > MIN_VAL: return 0
    
    max_alpha = 10
    if grad_val > 1e-6:
        max_alpha = min(max_alpha, (param_val-MIN_VAL)/grad_val)
    elif grad_val < -1e-6:
        max_alpha = min(max_alpha, (MIN_VAL-param_val)/grad_val)

    if limit_to_1:
        if grad_val > 1e-6:
            max_alpha = min(max_alpha, (1-param_val-MIN_VAL)/grad_val)
        elif grad_val < -1e-6:
            max_alpha = min(max_alpha, (param_val+MIN_VAL-1)/grad_val)

    return max_alpha    

def coordinate_ascent(r1, r2, theta, gradient_magnitude, 
                      fix_mu=False, fix_sigma=False):
    for j in range(len(theta)):
        if fix_mu and j == 0: continue
        if fix_sigma and j == 1: continue
        
        prev_loss = calc_loss(r1, r2, theta)

        # find the direction of the gradient
        gradient = numpy.zeros(len(theta))
        gradient[j] = gradient_magnitude
        init_alpha = 5e-12
        while init_alpha < 1e-2:
            pos = calc_loss( r1, r2, theta - init_alpha*gradient )
            neg = calc_loss( r1, r2, theta + init_alpha*gradient )
            if neg < prev_loss < pos:
                gradient[j] = gradient[j]
                #assert(calc_loss( 
                #       r1, r2, theta - init_alpha*gradient ) > prev_loss)
                #assert(calc_loss( 
                #       r1, r2, theta + init_alpha*gradient ) <= prev_loss)
                break
            elif neg > prev_loss > pos:
                gradient[j] = -gradient[j]
                #assert(calc_loss( 
                #    r1, r2, theta - init_alpha*gradient ) > prev_loss)
                #assert(calc_loss( 
                #    r1, r2, theta + init_alpha*gradient ) <= prev_loss)
                break
            else:
                init_alpha *= 10         

        #print( pos - prev_loss, neg - prev_loss )
        assert init_alpha < 1e-1
        
        min_step = 0
        max_step = find_max_step_size(
            theta[j], gradient[j], (False if j in (0,1) else True))

        if max_step < 1e-12: continue

        alpha = fminbound(
            lambda x: calc_loss( r1, r2, theta + x*gradient ),
            min_step, max_step)
        
        
        loss = calc_loss( r1, r2, theta + alpha*gradient )
        #print( "LOSS:", loss, prev_loss, loss-prev_loss )
        if loss < prev_loss:
            theta += alpha*gradient

    return theta

def gradient_ascent(r1, r2, theta, gradient_magnitude, 
                    fix_mu=False, fix_sigma=False):
    for j in range(len(theta)):
        if fix_mu and j == 0: continue
        if fix_sigma and j == 1: continue
        
        prev_loss = calc_loss(r1, r2, theta)

        mu, sigma, rho, p = theta
        z1 = compute_pseudo_values(r1, mu, sigma, p)
        z2 = compute_pseudo_values(r2, mu, sigma, p)
        real_grad = calc_pseudo_log_lhd_gradient(theta, z1, z2, False, False)
        
        gradient = numpy.zeros(len(theta))
        gradient[j] = gradient_magnitude
        if real_grad[j] < 0: gradient[j] = -gradient[j]
                
        min_step = 0
        max_step = find_max_step_size(
            theta[j], gradient[j], (False if j in (0,1) else True))

        if max_step < 1e-12: continue

        alpha = fminbound(
            lambda x: calc_loss( r1, r2, theta + x*gradient ),
            min_step, max_step)
                
        loss = calc_loss( r1, r2, theta + alpha*gradient )
        if loss < prev_loss:
            theta += alpha*gradient

    return theta


def find_local_maximum_CA(r1, r2, theta, 
                          fix_mu=False, fix_sigma=False ):
    gradient_magnitude = 1e-2
    for i in range(100):
        prev_loss = calc_loss(r1, r2, theta)
        
        # coordiante ascent step
        theta = coordinate_ascent( r1, r2, theta, gradient_magnitude,
                                   fix_mu=fix_mu, fix_sigma=fix_sigma)

        curr_loss = calc_loss(r1, r2, theta)
        if VERBOSE:
            print( "CA%i\t" % i, 
                   "%.2e" % gradient_magnitude, 
                   "%.2e" % (curr_loss-prev_loss), 
                   "%.8f\t" % curr_loss,
                   "%.8f\t" % log_lhd_loss(r1, r2, theta),
                   theta)

        # find the em estimate 
        mu, sigma, rho, p = theta
        z1 = compute_pseudo_values(r1, mu, sigma, p)
        z2 = compute_pseudo_values(r2, mu, sigma, p)
        em_theta = EM_step(z1, z2, theta )

        for j in (3,2,1,0):
            tmp_theta = theta.copy()
            tmp_theta[j] = em_theta[j]
            if calc_loss(r1, r2, tmp_theta) < curr_loss:
                theta[j] = em_theta[j]
        
        mu, sigma, rho, p = theta
        z1 = compute_pseudo_values(r1, mu, sigma, p)
        z2 = compute_pseudo_values(r2, mu, sigma, p)
        grad = calc_pseudo_log_lhd_gradient(theta, z1, z2, False, False)
        #print( "GRAD", grad )

        if abs(curr_loss-prev_loss) < 1e-12:
            if gradient_magnitude > 1e-6:
                gradient_magnitude /= 3
            else:
                return ( theta, curr_loss )
        else:
            gradient_magnitude = min(1e-2, gradient_magnitude*10)
        
    return theta, curr_loss

def find_local_maximum_PV(r1, r2, theta, N=100, EPS=1e-6,
                          fix_mu=False, fix_sigma=False ):
    for i in range(N):
        prev_loss = calc_loss(r1, r2, theta)
        curr_loss = prev_loss
        
        # find the em estimate 
        mu, sigma, rho, p = theta
        z1 = compute_pseudo_values(r1, mu, sigma, p)
        z2 = compute_pseudo_values(r2, mu, sigma, p)
        em_theta = EM_step(z1, z2, theta )

        # take a step in the EM direction
        for j in (3,2,1,0):
            tmp_theta = theta.copy()
            tmp_theta[j] = em_theta[j]
            new_loss = calc_loss(r1, r2, tmp_theta)
            if new_loss < curr_loss:
                theta[j] = em_theta[j]
                curr_loss = new_loss
        
        if VERBOSE:
            print( "CA%i\t" % i, 
                   "%.2e" % gradient_magnitude, 
                   "%.2e" % (curr_loss-prev_loss), 
                   "%.8f\t" % curr_loss,
                   "%.8f\t" % log_lhd_loss(r1, r2, theta),
                   theta)

        
        mu, sigma, rho, p = theta
        z1 = compute_pseudo_values(r1, mu, sigma, p)
        z2 = compute_pseudo_values(r2, mu, sigma, p)
        grad = calc_gaussian_mix_log_lhd_gradient(theta, z1, z2, False, False)
        
        if abs(curr_loss-prev_loss) < EPS:
            return ( theta, curr_loss )
    
    return theta, curr_loss


def clip_model_params(init_theta):
    theta = init_theta.copy()
    if theta[0] < MIN_MU:
        theta[0] = MIN_MU

    if theta[1] < MIN_SIGMA:
        theta[1] = MIN_SIGMA

    if theta[2] < MIN_RHO:
        theta[2] = MIN_RHO
    elif theta[2] > MAX_RHO:
        theta[2] = MAX_RHO

    if theta[3] < MIN_MIX_PARAM:
        theta[3] = MIN_MIX_PARAM
    elif theta[3] > MAX_MIX_PARAM:
        theta[3] = MAX_MIX_PARAM
        
    return theta, not (theta == init_theta).all()

def EM_iteration(z1, z2, prev_theta, max_iter,
                 fix_mu=False, fix_sigma=False, eps=1e-12):
    """Fit the gaussian model params via EM.

    """
    init_lhd = calc_gaussian_mix_log_lhd(prev_theta, z1, z2)
    prev_lhd = init_lhd
    for i in range(max_iter):
        theta = EM_step(
            z1, z2, prev_theta, fix_mu=fix_mu, fix_sigma=fix_sigma)
        theta, changed_params = clip_model_params(theta)
        
        new_lhd = calc_gaussian_mix_log_lhd(theta, z1, z2)
        if not changed_params:
            if not fix_sigma and not fix_mu:
                assert new_lhd + 1e-6 >= prev_lhd
            if new_lhd - prev_lhd < eps:
                return theta, new_lhd, i+1
        
        prev_theta = theta
        prev_lhd = new_lhd
    
    return theta, new_lhd, i+1


def EMP_with_pseudo_value_algorithm(
        r1, r2, theta_0, 
        N=100, EPS=1e-4, 
        fix_mu=False, fix_sigma=False):
    theta = theta_0
    z1 = compute_pseudo_values(r1, theta[0], theta[1], theta[3])
    z2 = compute_pseudo_values(r2, theta[0], theta[1], theta[3])

    max_num_EM_iter = 30
    
    for i in range(N):
        prev_theta = theta
        theta, new_lhd, num_EM_iter = EM_iteration(
            z1, z2, prev_theta, max_num_EM_iter, 
            fix_mu=fix_mu, fix_sigma=fix_sigma, eps=EPS/10)
        
        sum_param_change = numpy.abs(theta - prev_theta).sum()

        prev_z1 = z1
        z1 = compute_pseudo_values(r1, theta[0], theta[1], theta[3])
        prev_z2 = z2
        z2 = compute_pseudo_values(r2, theta[0], theta[1], theta[3])
        mean_pseudo_val_change = (
            numpy.abs(prev_z1-z1).mean() + numpy.abs(prev_z2-z2).mean())
        
        if VERBOSE:
            print( ("Iter %i" % i).ljust(12), 
                   "%.2e" % sum_param_change,
                   "%.2e" % mean_pseudo_val_change,
                   #"%.4e" % log_lhd_loss(r1, r2, theta),
                   theta)
        
        if i > 3 and (sum_param_change < EPS and mean_pseudo_val_change < EPS): 
            if num_EM_iter >= max_num_EM_iter and max_num_EM_iter < 1000:
                max_num_EM_iter *= 2
                if VERBOSE:
                    print( "WARNING: params appear to have converged but EM hasn't converged - increasing number of EM iterations to %i (this may just mean that 1 of the parameters converged to a boundary)" % max_num_EM_iter )
            else:
                break
    
    return theta, log_lhd_loss(r1, r2, theta)


def estimate_model_params(
        r1, r2, 
        theta_0, 
        max_iter=5000, convergence_eps=1e-10, 
        fix_mu=False, fix_sigma=False):

    theta, loss = EMP_with_pseudo_value_algorithm(
        r1, r2, theta_0, N=max_iter, EPS=convergence_eps, 
        fix_mu=fix_mu, fix_sigma=fix_sigma)
    
    return theta, loss

def main():
    r1_values, r2_values = [], []
    with open(sys.argv[1]) as fp:
        for line in fp:
            r1, r2, _ = line.split()
            r1_values.append(float(r1))
            r2_values.append(float(r2))
    r1_ranks, r2_ranks = [(-numpy.array(x)).argsort().argsort() 
                          for x in (r1_values, r2_values)]
    print( "Finished Loading Data" )
    
    starting_point = numpy.array( params )
    theta, log_lhd = estimate_model_params(
        r1_ranks, r2_ranks, starting_point, 
        fix_mu=False, fix_sigma=False)
        
    print( "NEW", theta, log_lhd)
    
    return


if __name__ == '__main__':
    main()

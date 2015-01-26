import os, sys

import math

import numpy

from scipy.special import erf, erfinv
from scipy.optimize import bisect, brentq

import timeit

import pyximport; pyximport.install()
from utility import (
    py_compute_pseudo_values, compute_pseudo_values, 
    simulate_values, py_cdf, py_cdf_i )
from inv_cdf import cdf, cdf_i, cdf_d1, c_compute_pseudo_values, c_compute_pseudo_values_old
def compute_pseudo_values(r, mu, sigma, rho):
    z = numpy.zeros(len(r), dtype=float)
    res = c_compute_pseudo_values(r, z, mu, sigma, rho)
    return res

def compute_pseudo_values_old(r, mu, sigma, rho):
    z = numpy.zeros(len(r), dtype=float)
    res = c_compute_pseudo_values_old(r, z, mu, sigma, rho)
    return res


def symbolic_computations():
    """Infrastructure to calculate derivatices and integrals. 

    This just does the algebra for us - I keep it here in case we want to
    modify things or think that there may be an error. 
    
    Currently it's printing the first and second derivates of the inverse cdf. 
    """
    import sympy
    class GaussianPDF(sympy.Function):
        nargs = 3
        is_commutative = False

        @classmethod
        def eval(cls, x, mu, sigma):
            std_x = (x - mu)/sigma    
            return ((
                1/(sigma*sympy.sqrt(sympy.pi*2))
            )*sympy.exp(-(std_x**2)/2))

    class GaussianMixturePDF(sympy.Function):
        nargs = 4

        @classmethod
        def eval(cls, x, mu, sigma, lamda):
            return (1-lamda)*GaussianPDF(x, 0, 1) + lamda*GaussianPDF(x, mu, sigma)

    class GaussianMixtureCDF(sympy.Function):
        nargs = 4

        res_str = "lamda*erf(sqrt(2)*tmp/2)/2 + lamda*erf(sqrt(2)*(mu - tmp)/(2*sigma))/2 - erf(sqrt(2)*(mu - tmp)/(2*sigma))/2 + 1/2"

        @classmethod
        def eval(cls, x, mu, sigma, lamda):
            try:
                assert False
                with open("cached_gaussian_mix_cdf.obj") as fp:
                    rv = pickle.load(fp)
            except:
                z = sympy.symbols("z", real=True, finite=True)
                rv = sympy.simplify(sympy.Integral(
                        GaussianMixturePDF(z, mu, sigma, lamda), 
                        (z, -sympy.oo, x)).doit())
                #with open("cached_gaussian_mix_cdf.obj", "w") as fp:
                #    rv = pickle.dump(rv, fp)                
            return rv

    lamda, sigma = sympy.symbols(
        "lamda, sigma", positive=True, real=True, finite=True)
    mu, rho = sympy.symbols(
        "mu, rho", real=True, finite=True)
    z = sympy.symbols("z", real=True, finite=True)

    r = GaussianMixtureCDF(z, mu, sigma, lamda)

    sympy.pprint( r )

    sympy.pprint( r.diff(z) )
    print( r.diff(z) )

    sympy.pprint( r.diff(z).diff(z) )
    print( r.diff(z).diff(z) )

    return

################################################################################
#
# Code to calculate derivatives for higher order convergence
#

def FD_d1(x, mu, sigma, lamda):
    return (py_cdf(x+1e-6, mu, sigma, lamda) - py_cdf(x-1e-6, mu, sigma, lamda))/2e-6

def py_cdf_d1(x, mu, sigma, lamda):
    pi = 3.14159265358979323846264338327950288419716939937510582
    pre = 1./math.sqrt(2*pi)

    noise = (1-lamda)*math.exp(-0.5*(x**2))

    norm_x = (x - mu)/sigma
    signal = lamda*math.exp(-0.5*(norm_x**2))
    return pre*(signal + noise)

def py_cdf_d1_simple(x, mu, sigma, lamda):
    pi = 3.14159265358979323846264338327950288419716939937510582
    return -math.sqrt(2)*lamda*math.exp(-x**2/2)/(2*math.sqrt(pi)) \
        + math.sqrt(2)*lamda*math.exp(-(mu - x)**2/(2*sigma**2))/(2*math.sqrt(pi)*sigma) \
        + math.sqrt(2)*math.exp(-x**2/2)/(2*math.sqrt(pi))

def py_cdf_d1_and_2(x, mu, sigma, lamda):
    pi = 3.14159265358979323846264338327950288419716939937510582
    pre = 1./math.sqrt(2*pi)

    noise = (1-lamda)*math.exp(-0.5*(x**2))

    norm_x = (x - mu)/sigma
    signal = lamda*math.exp(-0.5*(norm_x**2))
    
    d1 = pre*(signal + noise)
    d2 = -pre*(x*noise + (norm_x/(sigma**2))*signal)
    return d1, d2

def test_deriv():
    for x in range(10):
        print( py_cdf_d1_simple( x, 0, 1, 0.5 ) )
        print( py_cdf_d1( x, 0, 1, 0.5 ) )
        print( cdf_d1( x, 0, 1, 0.5 ))
        print( FD_d1( x, 0, 1, 0.5 ) )
        print()

################################################################################
#
# Fixed point method functions
#

def nm_step(x, mu, sigma, lamda, r):
    f = py_cdf(x, mu, sigma, lamda) - r
    d = 1e-12 + py_cdf_d1(x, mu, sigma, lamda)
    #print x, f, d, f/d
    return x - f/max(0.1, d)

def halley_step(x, mu, sigma, lamda, r):
    f = py_cdf(x, mu, sigma, lamda) - r
    d1, d2 = py_cdf_d1_and_2(x, mu, sigma, lamda)

    num = 2*f*d1
    denom = 2*d1*d1 - f*d2
    #print "Halley", num, denom, "d1", d1, "f", f, "d2", d2
    return x - num/(5*denom)


def main():
    mu, sigma, lamda = 1, 1, 0.9
    r = 1e-1
    new_x, x = -1.2, 1e9
    i = 0
    while i < 50 and abs(x - new_x) > 1e-6:
        x = new_x
        new_x = halley_step(x, mu, sigma, lamda, r)
        print( "H", abs(x-new_x), x, new_x )
        i += 1

    print( r-cdf( x, mu, sigma, lamda ), r, cdf( x, mu, sigma, lamda ) )

main()
assert False

params = (1, 1, 0.9, 0.5)
(r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(
    10000, params)

def t1():
    return compute_pseudo_values_old(r1_ranks, 1, 1, 0.5)

def t2():
    return compute_pseudo_values(r1_ranks, 1, 1, 0.5)
    
#print( timeit.timeit( "t1()", number=10, setup="from __main__ import t1"  ) )

#print( timeit.timeit( "t2()", number=10, setup="from __main__ import t2"  ) )

#test_i_cdf()

#test_deriv()
#main()

#symbolic_computations()

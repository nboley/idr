cimport cython
cimport numpy as np
from libc.math cimport exp, sqrt, pow, log, erf, fabs

cdef inline double d_max(double a, double b): return a if a >= b else b
cdef inline double d_min(double a, double b): return a if a <= b else b

cdef double EPS = 1e-8

@cython.cdivision(True)
cdef double c_cdf(double x, double mu, double sigma, double lamda):
    norm_x = (x-mu)/sigma
    return 0.5*lamda*erf(norm_x/sqrt(2)) + 0.5*(1-lamda)*erf(x/sqrt(2)) + 0.5

@cython.cdivision(True)
cdef double c_cdf_d1(double x, double mu, double sigma, double lamda):
    cdef double pi = 3.14159265358979323846264338327950288419716939937510582
    cdef double pre = 1./sqrt(2*pi)

    cdef double noise = (1-lamda)*exp(-0.5*(x**2))

    cdef double norm_x = (x - mu)/sigma
    cdef double signal = lamda*exp(-0.5*(norm_x**2))
    return pre*(signal + noise)

@cython.cdivision(True)
def cdf_d1(double x, double mu, double sigma, double lamda):
    return c_cdf_d1(x, mu, sigma, lamda)

@cython.cdivision(True)
def cdf(double x, double mu, double sigma, double lamda):
    return c_cdf(x, mu, sigma, lamda)

@cython.cdivision(True)
cdef double c_cdf_i(double r, double mu, double sigma, double lamda, 
                    double lb, double ub):
    for i in range(1000):
        mid = lb + (ub - lb)/2.;
        guess = c_cdf(mid, mu, sigma, lamda)
        if fabs(guess - r) < EPS:
            return mid
        elif guess < r:
            lb = mid
        else:
            ub = mid
    
    return mid

@cython.cdivision(True)
def cdf_i(double r, double mu, double sigma, double lamda, 
          double lb, double ub):
    lb = d_min(0, mu) - 10/d_min(1, sigma)
    ub = d_max(0, mu) + 10/d_min(1, sigma)

    while c_cdf(lb, mu, sigma, lamda) > r:
        lb -= 10
    
    while c_cdf(ub, mu, sigma, lamda) < r:
        ub += 10

    cdef double res = c_cdf_i(r, mu, sigma, lamda, lb, ub)
    if c_cdf(res, mu, sigma, lamda) - r < EPS:
        return res
    
    assert False

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def c_compute_pseudo_values_old(
        np.ndarray[np.int_t, ndim=1] rs, 
        np.ndarray[np.double_t, ndim=1] zs, 
        double mu, double sigma, double lamda ):
    cdef int N = len(rs)
    cdef double pseudo_N = N+1
    cdef double smallest_r = 1./pseudo_N
    lb = d_min(0, mu)
    while c_cdf(lb, mu, sigma, lamda) > smallest_r:
        lb -= 1

    ub = d_max(0, mu)
    while c_cdf(ub, mu, sigma, lamda) < 1-smallest_r:
        ub += 1

    for i in range(N):
        zs[i] = c_cdf_i((rs[i]+1)/pseudo_N, mu, sigma, lamda, lb, ub)
    
    return zs

from libc.stdlib cimport malloc, free

@cython.cdivision(True)
cdef void c_cdf_d1_and_2(double x, double mu, double sigma, double lamda,
                          double* d1, double* d2):
    pi = 3.14159265358979323846264338327950288419716939937510582
    pre = 1./sqrt(2*pi)

    noise = (1-lamda)*exp(-0.5*(x**2))

    norm_x = (x - mu)/sigma
    signal = lamda*exp(-0.5*(norm_x**2))
    
    d1[0] = pre*(signal + noise)
    d2[0] = -pre*(x*noise + (norm_x/(sigma**2))*signal)
    return

@cython.cdivision(True)
cdef double halley_step(double x, 
                        double mu, double sigma, double lamda, 
                        double r):
    cdef double f = c_cdf(x, mu, sigma, lamda) - r
    cdef double d1 = 0
    cdef double d2 = 0
    c_cdf_d1_and_2(x, mu, sigma, lamda, &d1, &d2)
    cdef double num = 2*f*d1
    cdef double denom = 2*d1*d1 - f*d2
    #print "Halley", num, denom, "d1", d1, "f", f, "d2", d2
    return x - num/(10*denom)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def c_compute_pseudo_values(
        np.ndarray[np.int_t, ndim=1] rs, 
        np.ndarray[np.double_t, ndim=1] zs, 
        double mu, double sigma, double lamda ):
    cdef int N = len(rs)
    cdef double pseudo_N = N+1
    cdef double* ordered_zs = <double * >malloc(N*sizeof(double))
    
    cdef double smallest_r = 1./pseudo_N
    lb = d_min(0, mu)
    while c_cdf(lb, mu, sigma, lamda) > smallest_r:
        lb -= 1

    ub = d_max(0, mu)
    while c_cdf(ub, mu, sigma, lamda) < 1-smallest_r:
        ub += 1
    
    lb = c_cdf_i(smallest_r, mu, sigma, lamda, lb, ub)
    ordered_zs[0] = lb
    
    cdef size_t i = 0, j= 0
    cdef double r = 0
    cdef double res = 10, prev_res = 10
    for i in range(1, N):
        r = (i+1)/pseudo_N
        if c_cdf(ub, mu, sigma, lamda) < r:
            ub += 10
        res = c_cdf_i(r, mu, sigma, lamda, lb, ub)
        
        ordered_zs[i] = res
        lb = res
        ub = lb + 2*(res - ordered_zs[i-1])
    
    for i in range(N):
        zs[i] = ordered_zs[rs[i]]
    free( ordered_zs )
    
    return zs

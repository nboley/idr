cimport cython
cimport numpy as np
from libc.math cimport exp, sqrt, pow, log, erf, fabs
from libc.stdlib cimport malloc, free

cdef inline double d_max(double a, double b): return a if a >= b else b
cdef inline double d_min(double a, double b): return a if a <= b else b

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
                    double lb, double ub, double EPS):
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
          double lb, double ub, double EPS):
    lb = d_min(0, mu) - 10/d_min(1, sigma)
    ub = d_max(0, mu) + 10/d_min(1, sigma)

    while c_cdf(lb, mu, sigma, lamda) > r:
        lb -= 10
    
    while c_cdf(ub, mu, sigma, lamda) < r:
        ub += 10

    cdef double res = c_cdf_i(r, mu, sigma, lamda, lb, ub, EPS)
    if c_cdf(res, mu, sigma, lamda) - r < EPS:
        return res
    
    assert False

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def c_compute_pseudo_values(
        np.ndarray[np.int_t, ndim=1] rs, 
        np.ndarray[np.double_t, ndim=1] zs, 
        double mu, double sigma, double lamda,
        double EPS):
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
    
    lb = c_cdf_i(smallest_r, mu, sigma, lamda, lb, ub, EPS)
    ordered_zs[0] = lb
    
    cdef size_t i = 0, j= 0
    cdef double r = 0
    cdef double res = 10, prev_res = 10
    for i in range(1, N):
        r = (i+1)/pseudo_N
        if c_cdf(ub, mu, sigma, lamda) < r:
            ub += 10
        res = c_cdf_i(r, mu, sigma, lamda, lb, ub, EPS)
        
        ordered_zs[i] = res
        lb = res
        ub = lb + 2*(res - ordered_zs[i-1])
    
    for i in range(N):
        zs[i] = ordered_zs[rs[i]]
    free( ordered_zs )
    
    return zs

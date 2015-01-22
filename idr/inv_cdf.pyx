cimport cython
from libc.math cimport exp, sqrt, pow, log, erf, abs

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
def cdf_i(double r, double mu, double sigma, double lamda, 
          double lb, double ub):
    lb = d_min(0, mu) - 10/d_min(1, sigma)
    ub = d_max(0, mu) + 10/d_min(1, sigma)

    while c_cdf(lb, mu, sigma, lamda) > r:
        lb -= 10
    
    while c_cdf(ub, mu, sigma, lamda) < r:
        ub += 10
        
    for i in range(1000):
        mid = lb + (ub - lb)/2.;
        guess = c_cdf(mid, mu, sigma, lamda)
        if abs(guess - r) < 1e-12:
            return mid
        elif guess < r:
            lb = mid
        else:
            ub = mid

    assert False

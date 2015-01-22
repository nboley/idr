import numpy

import sympy
from sympy.printing.theanocode import theano_function

def build_mixture_loss_and_grad(build_pseudo_functions=False):
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
            return ( (1-lamda)*GaussianPDF(x, 0, 1) 
                     + lamda*GaussianPDF(x, mu, sigma) )

    class GaussianMixtureCDF(sympy.Function):
        nargs = 4
        
        @classmethod
        def eval(cls, x, mu, sigma, lamda):
            z = sympy.symbols("z", real=True, finite=True)
            rv = sympy.simplify(sympy.Integral(
                    GaussianMixturePDF(z, mu, sigma, lamda), 
                    (z, -sympy.oo, x)).doit())
            return rv
        
    class GaussianMixtureCDF_inverse(sympy.Function):
        """
        @classmethod
        def eval(cls, r, mu, sigma, lamda):
            if mu == 0 and sigma == 1:
                return sympy.erfi(r)
            return sympy.Function('GaussianMixtureCDF_inverse')(
                r, mu, sigma, lamda)
        """
        def _eval_is_real(self):
            return True

        def _eval_is_finite(self):
            return True

        def fdiff(self, argindex):
            r, mu, sigma, lamda = self.args
            # if mu=0 and sigma=1, then this is
            # just the inverse standard erf so return erfi
            if mu == 0 and sigma == 1:
                return sympy.diff(sympy.erfi(r), self.args[argindex-1])

            tmp = sympy.symbols("tmp", real=True, finite=True)
            z_s = GaussianMixtureCDF(tmp, mu, sigma, lamda)
            inv_diff = sympy.diff(z_s, self.args[argindex-1])
            return sympy.simplify(1/inv_diff.subs(tmp, self))            

    # create symbols for the modle params
    lamda_s, sigma_s = sympy.symbols(
        "lamda, sigma", positive=True, real=True, finite=True)
    mu_s, rho_s = sympy.symbols(
        "mu, rho", real=True, finite=True)

    # if we are building the pseudo functiosn then
    # the ranks are what we actually observe, so we need 
    # to wrap them in an inverse CDF call
    if build_pseudo_functions:
        r1_s = sympy.symbols("r1_s", real=True, finite=True, positive=True)
        r2_s = sympy.symbols("r2_s", real=True, finite=True, positive=True)
        z1_s = GaussianMixtureCDF_inverse(r1_s, mu_s, sigma_s, lamda_s)
        z2_s = GaussianMixtureCDF_inverse(r2_s, mu_s, sigma_s, lamda_s)
    # otherwise we just use standard symbols for the obsreved values
    else:
        z1_s, z2_s = sympy.symbols("z1, z2", real=True, finite=True)

    ####### build the marginal densities    
    std_z1_s = (z2_s - mu_s)/sigma_s
    std_z2_s = (z1_s - mu_s)/sigma_s

    ####### bivariate normal density
    sym_signal_density = (
                       1./(2.*sympy.pi*sigma_s*sigma_s)
                      )*(
                       1./sympy.sqrt(1.-rho_s**2)
                      )*sympy.exp(-(
                          std_z1_s**2 + std_z2_s**2 - 2*rho_s*std_z1_s*std_z2_s
                      )/(2*(1-rho_s**2)))

    sym_noise_density = (
                       1./(2.*sympy.pi)
                      )*sympy.exp(-(z1_s**2 + z2_s**2)/2)

    sym_log_lhd = sympy.simplify(sympy.log(lamda_s*sym_signal_density 
                                           + (1-lamda_s)*sym_noise_density))

    # we use the following in the theano calls instead of the z's
    # so that theano won't choke
    pv_1, pv_2 = sympy.symbols('pv_1 pv_2', real=True, finite=True)

    # differentiate, replace the inverse micture CDF's with pv_'s,
    # and then build the theano functions
    sym_gradients = []
    for sym in (mu_s, sigma_s, rho_s, lamda_s):
        sym_grad = sympy.diff(sym_log_lhd, sym)
        pv_sym_grad = sym_grad.subs({z1_s: pv_1, z2_s: pv_2})
        sym_gradients.append( pv_sym_grad )

    theano_gradient = theano_function(
        (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
        sym_gradients,
        dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})    

    theano_log_lhd = theano_function(
        (mu_s, sigma_s, rho_s, lamda_s, pv_1, pv_2), 
        [sym_log_lhd.subs({z1_s: pv_1, z2_s: pv_2}),],
        dims={mu_s:1, sigma_s:1, rho_s:1, lamda_s:1, pv_1: 1, pv_2:1})    
        
    # wrap the theano functions in python functions, and return them
    def calc_log_lhd(theta, z1, z2):
        mu, sigma, rho, lamda = theta
        
        return theano_log_lhd(
            numpy.repeat(mu, len(z1)),
            numpy.repeat(sigma, len(z1)),
            numpy.repeat(rho, len(z1)),
            numpy.repeat(lamda, len(z1)),
            z1, z2 ).sum()
    
    def calc_log_lhd_gradient(theta, z1, z2, fix_mu, fix_sigma):
        mu, sigma, rho, lamda = theta

        res = theano_gradient(
            numpy.repeat(mu, len(z1)),
            numpy.repeat(sigma, len(z1)),
            numpy.repeat(rho, len(z1)),
            numpy.repeat(lamda, len(z1)),
            z1, z2 )
        return numpy.array( [x.sum() for x in res] )
    
    return calc_log_lhd, calc_log_lhd_gradient

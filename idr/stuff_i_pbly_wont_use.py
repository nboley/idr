def cdf_and_inv_cdf_gen(mu, sigma, pi, min_val=-100, max_val=100):
    def cdf(x):
        norm_x = (x-mu)/sigma
        return 0.5*(1-pi)*erf(norm_x/math.sqrt(2)) + 0.5*pi*erf(x/math.sqrt(2)) + 0.5
    
    def inv_cdf(r, start=min_val, stop=max_val):
        assert r > 0 and r < 1
        return brentq(lambda x: cdf(x) - r, min_val, max_val)
    
    return cdf, inv_cdf


def compute_pseudo_values(ranks, signal_mu, signal_sd, p):
    cdf, inv_cdf = cdf_and_inv_cdf_gen(signal_mu, signal_sd, p, -20, 20)
    pseudo_values = []
    for x in ranks:
        new_x = float(x+1)/(len(ranks)+1)
        pseudo_values.append( inv_cdf( new_x ) )

    return numpy.array(pseudo_values)


def compute_pseudo_values_COMPARE_METHODS(ranks, signal_mu, signal_sd, p, 
                                          NB=100):
    norm_ranks = (ranks+1.)/(len(ranks)+1.)

    cdf, inv_cdf = cdf_and_inv_cdf_gen(signal_mu, signal_sd, p)

    #print( cdf(inv_cdf(norm_ranks.max()) ))
    #return
    min_val = inv_cdf(norm_ranks.min())
    max_val = inv_cdf(norm_ranks.max())

    # build an evenly space grid across the values
    values = numpy.linspace(min_val, max_val, num=NB)
    cdf = numpy.array([cdf(x) for x in values])
    cdf[0] -= 1e-6
    cdf[-1] += 1e-6

    print( cdf )
    pseudo_values = []
    for x in norm_ranks:
        i = cdf.searchsorted(x)
        start, stop = values[i-1], values[i]
        pseudo_values.append( inv_cdf(x, start, stop) )
        continue
        #yield new_x, values[i]
        #print( new_x, values[i-1], values[i], values[i+1] )
        
        print(x, i)
        print( x, (cdf[i-1], cdf[i]), (values[i-1], values[i]),
               inv_cdf( x ), math.sqrt(2)*erfinv(2*x-1) )

        assert cdf[i-1] <= x <= cdf[i]
        assert values[i-1] <= inv_cdf( x ) <= values[i]

               
               
        #pseudo_values.append(values[i])

    return #numpy.array(pseudo_values)

def compute_pseudo_values_grid_start(ranks, signal_mu, signal_sd, p, NB=100):
    norm_ranks = (ranks+1.)/(len(ranks)+1.)

    cdf, inv_cdf = cdf_and_inv_cdf_gen(signal_mu, signal_sd, p)

    min_val = inv_cdf(norm_ranks.min())
    max_val = inv_cdf(norm_ranks.max())

    # build an evenly space grid across the values
    values = numpy.linspace(min_val, max_val, num=NB)
    cdf = numpy.array([cdf(x) for x in values])
    # correct for rounding error
    cdf[0] -= 1e-6
    cdf[-1] += 1e-6

    pseudo_values = []
    for x in norm_ranks:
        i = cdf.searchsorted(x)
        start, stop = values[i-1], values[i]
        pseudo_values.append( inv_cdf(x, start, stop) )
        
        #assert cdf[i-1] <= x <= cdf[i]
        #assert values[i-1] <= inv_cdf( x ) <= values[i]
    
    return numpy.array(pseudo_values)

def update_mixture_params_estimate_full(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd_2(
        i_mu[0], i_mu[1], i_sigma[0], i_sigma[1], i_rho, z1, z2)
    
    ez = i_p*numpy.exp(signal_log_lhd)/(
        i_p*numpy.exp(signal_log_lhd)+(1-i_p)*numpy.exp(noise_log_lhd))
    
    # just a small optimization
    ez_sum = ez.sum()
    
    p = ez_sum/len(ez)
    
    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    
    weighted_sum_sqs_1 = (ez*((z1-mu_1)**2)).sum()
    sigma_1 = math.sqrt(weighted_sum_sqs_1/ez_sum)

    weighted_sum_sqs_2 = (ez*((z2-mu_2)**2)).sum()
    sigma_2 = math.sqrt(weighted_sum_sqs_2/ez_sum)
    
    rho = 2*((ez*(z1-mu_1))*(z2-mu_2)).sum()/(
        weighted_sum_sqs_1 + weighted_sum_sqs_2)
    
    jnt_log_lhd = numpy.log(
        i_p*numpy.exp(signal_log_lhd) + (1-i_p)*numpy.exp(noise_log_lhd)).sum()
    #print( jnt_log_lhd, ((mu_1, mu_2), (sigma_1, sigma_2), rho, p) )
    
    return ((mu_1, mu_2), (sigma_1, sigma_1), rho, p), jnt_log_lhd

def update_mixture_params_estimate_fixed(z1, z2, starting_point):
    i_mu, i_sigma, i_rho, i_p = starting_point
    
    noise_log_lhd = compute_lhd_2(0,0, 1,1, 0, z1, z2)
    signal_log_lhd = compute_lhd_2(
        i_mu[0], i_mu[1], i_sigma[0], i_sigma[1], i_rho, z1, z2)    
    ez = i_p*numpy.exp(signal_log_lhd)/(
        i_p*numpy.exp(signal_log_lhd)+(1-i_p)*numpy.exp(noise_log_lhd))
    ez_sum = ez.sum()
    
    p = ez_sum/len(ez)

    mu_1 = (ez*z1).sum()/(ez_sum)
    mu_2 = (ez*z2).sum()/(ez_sum)
    mu = (mu_1 + mu_2)/2
    mu_1 = mu_2 = mu
    
    sigma_1, sigma_2, sigma = 1., 1., 1.
    rho = (ez*(z1-mu)*(z2-mu)).sum()/(ez_sum)
    
    noise_log_lhd = compute_lhd(0, 1, 0, z1, z2)
    signal_log_lhd = compute_lhd(mu, sigma, rho, z1, z2)
    jnt_log_lhd = numpy.log(
        p*numpy.exp(signal_log_lhd) + (1-p)*numpy.exp(noise_log_lhd)).sum()
    return ((mu, mu), (sigma, sigma), rho, p), jnt_log_lhd, jnt_log_lhd

def compute_lhd(mu, sigma, rho, z1, z2):
    # -1.837877 = -log(2)-log(pi)
    coef = -1.837877-2*math.log(sigma) - math.log(1-rho*rho)/2
    loglik = coef-0.5/((1-rho*rho)*(sigma*sigma))*(
        (z1-mu)**2 -2*rho*(z1-mu)*(z2-mu) + (z2-mu)**2)
    return loglik


def main2():
    def calc_log_lhd(theta, z1, z2):
        mu, sigma, rho, p = theta

        noise_log_lhd = compute_lhd(0,0, 1,1, 0, z1, z2)
        signal_log_lhd = compute_lhd(
            mu, mu, sigma, sigma, rho, z1, z2)

        log_lhd = numpy.log(p*numpy.exp(signal_log_lhd)
                            +(1-p)*numpy.exp(noise_log_lhd)).sum()

        return log_lhd


    r1_values, r2_values = [], []
    with open(sys.argv[1]) as fp:
        for line in fp:
            r1, r2, _ = line.split()
            r1_values.append(float(r1))
            r2_values.append(float(r2))
    r1_ranks, r2_ranks = [(-numpy.array(x)).argsort().argsort() 
                          for x in (r1_values, r2_values)]
    #params = (mu, sigma, rho, p)
    for i in range(1):
        #params = (1, 1, 0.5, 0.5)
        #(r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(
        #    1000, params)
        #starting_point = ((1,1), (1,1), 0.5, 0.5)

        #params = (2.6, 1.3, 0.8, 0.7)
        params = (1.195264, 0.6619979, 0.8867832, 0.7615457)
        starting_point = ((params[0],params[0]), 
                          (params[1],params[1]), 
                          params[2],
                          params[3] )

        """
        z1 = compute_pseudo_values(r1_ranks, 1, 1, 0.5)
        z2 = compute_pseudo_values(r2_ranks, 1, 1, 0.5)
        theta, log_lhd =  update_mixture_params_archive(
            z1, z2, starting_point)
        """
        r1 = r1_ranks
        r2 = r2_ranks
        
        #theta, log_lhd = em_gaussian(
        #    r1_ranks, r2_ranks, starting_point,
        #    False, False, False)
        r1 = (numpy.array(r1_ranks, dtype=float) + 1)/(len(r1_ranks)+1)
        r2 = (numpy.array(r2_ranks, dtype=float) + 1)/(len(r2_ranks)+1)
        #new_log_lhd = calc_log_lhd_new((theta[0][0], theta[1][0], theta[2], theta[3]), 
        #                               GMCDF_i(r1, theta[0][0], theta[1][0], theta[3]),
        #                               GMCDF_i(r2, theta[0][1], theta[1][1], theta[3]))
        #
        #print( "EM", (theta[0][0], theta[1][0], theta[2], theta[3]), log_lhd, new_log_lhd )

        """
        theta, log_lhd =  update_mixture_params_archive(
            r1_values, r2_values, starting_point)
        #r1 = (numpy.array(r1_ranks, dtype=float) + 1)/(len(r1_ranks)+1)
        #r2 = (numpy.array(r2_ranks, dtype=float) + 1)/(len(r2_ranks)+1)
        new_log_lhd = calc_log_lhd_new((theta[0][0], theta[1][0], theta[2], theta[3]), 
                                       GMCDF_i(r1, theta[0][0], theta[1][0], theta[3]),
                                       GMCDF_i(r2, theta[0][0], theta[1][0], theta[3]))
        print( "OLD", theta, log_lhd, new_log_lhd )
        """
        
        #continue
        theta, log_lhd = update_mixture_params_estimate(
            r1_ranks, r2_ranks, starting_point)
        
        
        z1 = GMCDF_i(r1, theta[0], theta[1], theta[3])
        z2 = GMCDF_i(r2, theta[0], theta[1], theta[3])

        print( "NEW", tuple(theta.tolist()), calc_log_lhd(theta, z1, z2), log_lhd )
               
        print()
        continue
        theta, log_lhd = em_gaussian(r1_ranks, r2_ranks, params, 
                                      True, False, False)
        print("EM", theta, log_lhd, new_log_lhd )

    return

    params = (1, 1, 0.9, 0.5)
    (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(100, params)
    r1 = (numpy.array(r1_ranks, dtype=float)+1)/(len(r1_ranks)+1)
    r2 = (numpy.array(r2_ranks, dtype=float)+1)/(len(r2_ranks)+1)
    (calc_log_lhd, calc_log_lhd_gradient 
     ) = symbolic.build_copula_mixture_loss_and_grad()
    print( calc_log_lhd(params, r1_values, r2_values).sum() )
    print( calc_log_lhd_gradient(params, r1_values, r2_values) )

    return

    import pylab
    #pylab.scatter(r1_values, r2_values)
    #pylab.scatter(r1_ranks, r2_ranks)
    #pylab.show()
    #return
    init_params = ((1,1), (1,1), 0.1, 0.9)
    params, log_lhd = em_gaussian(
        r1_ranks, r2_ranks, init_params)
    print(params, log_lhd)
    return

    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, init_params, True)
    print("\nEM", params, log_lhd)
    
    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, init_params, 
                                  False, True, True)
    print("\nGA", params, log_lhd)
    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, params, 
                                  False, False, True)
    print("\nGA", params, log_lhd)
    params, log_lhd = em_gaussian(r1_ranks, r2_ranks, params, 
                                  False, False, False)
    print("\nGA", params, log_lhd)

    return
    
    print( estimate_mixture_params(r1_values, r2_values, params) )
    return
    print
    print(sim_values)
    import pylab
    pylab.scatter(sim_values[:,0], sim_values[:,1])
    pylab.show()
    pass


def update_mixture_params_estimate_natural(r1, r2, starting_point, 
                                           fix_mu=False, fix_sigma=False ):
    #r1 = (numpy.array(r1, dtype=float) + 1)/(len(r1)+1)
    #r2 = (numpy.array(r2, dtype=float) + 1)/(len(r2)+1)

    #curr_z1 = GMCDF_i(r1, starting_point[0][0], starting_point[1][0], starting_point[3])
    #curr_z2 = GMCDF_i(r2, starting_point[0][0], starting_point[1][0], starting_point[3])
    
    eta1, eta2, rho, p = starting_point

    theta = numpy.array((eta1[0], eta2[0], rho, p))

    def bnd_calc_log_lhd(theta):
        eta1, eta2, rho, p = theta
        z1 = GMCDF_i(r1, eta1/eta2, 1./eta2, p)
        z2 = GMCDF_i(r2, eta1/eta2, 1./eta2, p)
        rv = calc_log_lhd_new(theta, z1, z2)
        return rv

    def bnd_calc_log_lhd_gradient(theta):
        eta1, eta2, rho, p = theta
        z1 = GMCDF_i(r1, eta1/eta2, 1./eta2, p)
        z2 = GMCDF_i(r2, eta1/eta2, 1./eta2, p)
        return calc_log_lhd_gradient_new(
            theta, z1, z2, fix_mu, fix_sigma)
    
    def find_max_step_size(theta, grad):
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu

        ## everything is greater than 0 constraint
        # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
        # gradient[i]*alpha > -theta[i]
        # if gradient[i] > 0:
        #     alpha < theta[i]/gradient[i]
        # elif gradient[i] < 0:
        #     alpha < -theta[i]/gradient[i]
        max_alpha = 100
        for param_val, grad_val in zip(theta, grad):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, param_val/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, -param_val/grad_val)
        
        ## correlation and mix param are less than 1 constraint
        # theta[3] + gradient[3]*alpha < 1
        # gradient[3]*alpha < 1 - theta[3]
        # if gradient[2] > 0:
        #     alpha < (1 - theta[3])/gradient[3]
        # elif gradient[2] < 0:
        #     alpha < (theta[3] - 1)/gradient[3]
        for param_val, grad_val in zip(theta[2:], grad[2:]):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, (1-param_val)/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, (param_val-1)/grad_val)
        
        return max_alpha
    
    prev_lhd = bnd_calc_log_lhd(theta)

    for i in range(10000):
        gradient = bnd_calc_log_lhd_gradient(theta)
        norm_gradient = gradient/numpy.abs(gradient).sum()
        max_step_size = find_max_step_size(theta, norm_gradient)
        
        def bnd_objective(alpha):
            new_theta = theta + alpha*norm_gradient
            rv = -bnd_calc_log_lhd( new_theta )
            print( rv )
            return rv
                
        alpha = fminbound(bnd_objective, 0, max_step_size)
        #alpha = min(1e-2, find_max_step_size(theta))
        log_lhd = -bnd_objective(alpha)
        
        #while alpha > 0 and log_lhd < prev_lhd:
        #    alpha /= 10
        #    log_lhd = -bnd_objective(alpha)

        print( alpha, theta + alpha*norm_gradient )
        print( gradient )
        print( log_lhd, (gradient**2).sum() )
        
        if abs(log_lhd-prev_lhd) < 10*EPS:
            return ( theta, log_lhd )                     
        else:
            theta += alpha*norm_gradient
            #print( alpha, theta, log_lhd, prev_lhd )
            prev_lhd = log_lhd
    
    assert False

def test_timing():
    params = (1, 1, 0.0, 0.5)
    (r1_ranks, r2_ranks), (r1_values, r2_values) = simulate_values(
        1000, params)

    def t1():
        return compute_pseudo_values_simple(r1_ranks, 1, 1, 0.5)

    def t2():
        return compute_pseudo_values(r1_ranks, 1, 1, 0.5)

def update_mixture_params_estimate_BAD(r1, r2, starting_point, 
                                   fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point

    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        rv = calc_log_lhd_new(theta, z1, z2)
        return rv

    def bnd_calc_log_lhd_gradient(theta):
        z1 = GMCDF_i(r1, mu[0], sigma[0], p)
        z2 = GMCDF_i(r2, mu[1], sigma[1], p)
        return calc_log_lhd_gradient_new(
            theta, z1, z2, fix_mu, fix_sigma)
    
    def find_max_step_size(theta, grad):
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu

        ## everything is greater than 0 constraint
        # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
        # gradient[i]*alpha > -theta[i]
        # if gradient[i] > 0:
        #     alpha < theta[i]/gradient[i]
        # elif gradient[i] < 0:
        #     alpha < -theta[i]/gradient[i]
        max_alpha = 10000
        for param_val, grad_val in zip(theta, grad):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, param_val/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, -param_val/grad_val)
        
        ## correlation and mix param are less than 1 constraint
        # theta[3] + gradient[3]*alpha < 1
        # gradient[3]*alpha < 1 - theta[3]
        # if gradient[2] > 0:
        #     alpha < (1 - theta[3])/gradient[3]
        # elif gradient[2] < 0:
        #     alpha < (theta[3] - 1)/gradient[3]
        for param_val, grad_val in zip(theta[2:], grad[2:]):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, (1-param_val)/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, (param_val-1)/grad_val)
        
        return max_alpha
    
    prev_lhd = bnd_calc_log_lhd(theta)

    inactive_set = []
    inactive_alphas = []
    for i in range(10000):
        gradient = bnd_calc_log_lhd_gradient(theta)
        for index in inactive_set:
            gradient[index] = 0
        gradient[ numpy.abs(gradient) != numpy.abs(gradient).max() ] = 0
        #print( numpy.argmax(numpy.abs(gradient)) )
        current_index = numpy.argmax(numpy.abs(gradient)) 
        
        norm_gradient = gradient/numpy.abs(gradient).sum()
        max_step_size = find_max_step_size(theta, norm_gradient)
        
        def bnd_objective(alpha):
            new_theta = theta + alpha*norm_gradient
            rv = -bnd_calc_log_lhd( new_theta )
            return rv
                
        alpha = fminbound(bnd_objective, 0, max_step_size)
        #alpha = min(1e-2, find_max_step_size(theta))
        log_lhd = -bnd_objective(alpha)
        
        if log_lhd <= prev_lhd:
            inactive_set.append( current_index )
            inactive_alphas.append( alpha )
            if len( inactive_set ) < 4:
                continue
        else:
            theta += alpha*norm_gradient
            prev_lhd = log_lhd

        if len( inactive_set ) == 4:
            print( inactive_set )
            print( inactive_alphas )
            inactive_set, inactive_alphas = [], []
        
        print( log_lhd, log_lhd-prev_lhd, alpha, max_step_size, current_index )
        print( "gradient", bnd_calc_log_lhd_gradient(theta  + alpha*norm_gradient ) )
        print( "params", theta + alpha*norm_gradient )
        print( "="*20 )

        
        if False and abs(log_lhd-prev_lhd) < 10*EPS:            
                return ( theta, log_lhd )
    
    assert False

def update_mixture_params_estimate_BAD2(r1, r2, starting_point, 
                                        fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point

    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_loss(theta):
        mu, sigma, rho, p = theta
        z1 = GMCDF_i(r1, mu, sigma, p)
        z2 = GMCDF_i(r2, mu, sigma, p)
        rv = calc_loss(theta, z1, z2)
        return rv

    def bnd_calc_grad(theta):
        z1 = GMCDF_i(r1, mu[0], sigma[0], p)
        z2 = GMCDF_i(r2, mu[1], sigma[1], p)
        rv = calc_grad(
            theta, z1, z2, fix_mu, fix_sigma)
        return rv
    
    def find_max_step_size(theta, grad):
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu

        ## everything is greater than 0 constraint
        # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
        # gradient[i]*alpha > -theta[i]
        # if gradient[i] > 0:
        #     alpha < theta[i]/gradient[i]
        # elif gradient[i] < 0:
        #     alpha < -theta[i]/gradient[i]
        max_alpha = 100
        for param_val, grad_val in zip(theta, grad):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, param_val/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, -param_val/grad_val)
        
        ## correlation and mix param are less than 1 constraint
        # theta[3] + gradient[3]*alpha < 1
        # gradient[3]*alpha < 1 - theta[3]
        # if gradient[2] > 0:
        #     alpha < (1 - theta[3])/gradient[3]
        # elif gradient[2] < 0:
        #     alpha < (theta[3] - 1)/gradient[3]
        for param_val, grad_val in zip(theta[2:], grad[2:]):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, (1-param_val)/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, (param_val-1)/grad_val)
        
        return max_alpha
    
    prev_lhd = bnd_calc_loss(theta)

    for i in range(10000):
        for j in range(len(theta)):
            gradient = bnd_calc_grad(theta)
            gradient[j] = 0.0
            #for k in xrange(len(theta)):
            #    if k != j: gradient[j] = 0.0
            norm_gradient = gradient/(100*numpy.abs(gradient).sum())

            max_step_size = find_max_step_size(theta, norm_gradient)

            def bnd_objective(alpha):
                new_theta = theta - alpha*norm_gradient
                rv = bnd_calc_log_lhd( new_theta )
                return rv

            alpha = fminbound(bnd_objective, -1e-6, max_step_size)
            #alpha = min(1e-2, find_max_step_size(theta))
            log_lhd = bnd_objective(alpha)

            if False and abs(log_lhd-prev_lhd) < 10*EPS:            
                return theta
            else:
                theta -= alpha*norm_gradient
                prev_lhd = log_lhd

            print( log_lhd, log_lhd-prev_lhd, alpha, max_step_size )
            print( "gradient", gradient )
            print( "params", theta )
            
            
            
            print( "="*20 )

            

            if False and abs(log_lhd-prev_lhd) < 10*EPS:            
                    return ( theta, log_lhd )
    
    assert False

def em_gaussian(ranks_1, ranks_2, params, 
                use_EM=False, fix_mu=False, fix_sigma=False):
    lhds = []
    param_path = []
    prev_lhd = None
    for i in range(MAX_NUM_PSUEDO_VAL_ITER):
        mu, sigma, rho, p = params

        z1 = compute_pseudo_values(ranks_1, mu[0], sigma[0], p)
        z2 = compute_pseudo_values(ranks_2, mu[1], sigma[1], p)

        if use_EM:
            params, log_lhd = update_mixture_params_estimate_full(
                z1, z2, params)
        else:
            params, log_lhd = update_mixture_params_archive(
                z1, z2, params) #, fix_mu, fix_sigma)
        
        print( i, log_lhd, params )
        lhds.append(log_lhd)
        param_path.append(params)

        #print( i, end=" ", flush=True) # params, log_lhd
        if prev_lhd != None and abs(log_lhd - prev_lhd) < 1e-4:
            return params, log_lhd
        prev_lhd = log_lhd

    raise RuntimeError( "Max num iterations exceeded in pseudo val procedure" )


def update_mixture_params_archive(z1, z2, starting_point, 
                                  fix_mu=False, fix_sigma=False ):
    mu, sigma, rho, p = starting_point

    theta = numpy.array((mu[0], sigma[0], rho, p))

    def bnd_calc_log_lhd(theta):
        return calc_log_lhd(theta, z1, z2)

    def bnd_calc_log_lhd_gradient(theta):
        return calc_log_lhd_gradient(
            theta, z1, z2, fix_mu, fix_sigma)
    
    def find_max_step_size(theta):
        # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu
        grad = bnd_calc_log_lhd_gradient(theta)
        ## everything is greater than 0 constraint
        # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
        # gradient[i]*alpha > -theta[i]
        # if gradient[i] > 0:
        #     alpha < theta[i]/gradient[i]
        # elif gradient[i] < 0:
        #     alpha < -theta[i]/gradient[i]
        max_alpha = 1
        for param_val, grad_val in zip(theta, grad):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, param_val/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, -param_val/grad_val)
        
        ## correlation and mix param are less than 1 constraint
        # theta[3] + gradient[3]*alpha < 1
        # gradient[3]*alpha < 1 - theta[3]
        # if gradient[2] > 0:
        #     alpha < (1 - theta[3])/gradient[3]
        # elif gradient[2] < 0:
        #     alpha < (theta[3] - 1)/gradient[3]
        for param_val, grad_val in zip(theta[2:], grad[2:]):
            if grad_val > 1e-6:
                max_alpha = min(max_alpha, (1-param_val)/grad_val)
            elif grad_val < -1e-6:
                max_alpha = min(max_alpha, (param_val-1)/grad_val)
        
        return max_alpha
    
    prev_lhd = bnd_calc_log_lhd(theta)

    for i in range(10000):
        gradient = bnd_calc_log_lhd_gradient(theta)
        #gradient = gradient/numpy.abs(gradient).sum()
        #print( gradient )
        def bnd_objective(alpha):
            new_theta = theta + alpha*gradient
            return -bnd_calc_log_lhd( new_theta )
                
        alpha = fminbound(bnd_objective, 0, find_max_step_size(theta))
        log_lhd = -bnd_objective(alpha)
        
        #alpha = min(1e-11, find_max_step_size(theta))
        #while alpha > 0 and log_lhd < prev_lhd:
        #    alpha /= 10
        #    log_lhd = -bnd_objective(alpha)
        
        if abs(log_lhd-prev_lhd) < EPS:
            return ( ((theta[0], theta[0]), 
                      (theta[1], theta[1]), 
                      theta[2], theta[3]), 
                     log_lhd )
        else:
            theta += alpha*gradient
            #print( "\t", log_lhd, prev_lhd, theta )
            prev_lhd = log_lhd
    
    assert False

def full_find_max_step_size(theta, grad):
    # contraints: 0<p<1, 0<rho, 0<sigma, 0<mu

    ## everything is greater than 0 constraint
    # theta[i] + gradient[i]*alpha >>>> f = ufuncify([x], expr) 0
    # gradient[i]*alpha > -theta[i]
    # if gradient[i] > 0:
    #     alpha < theta[i]/gradient[i]
    # elif gradient[i] < 0:
    #     alpha < -theta[i]/gradient[i]
    max_alpha = 1
    for param_val, grad_val in zip(theta, grad):
        if grad_val > 1e-6:
            max_alpha = min(max_alpha, param_val/grad_val)
        elif grad_val < -1e-6:
            max_alpha = min(max_alpha, -param_val/grad_val)

    ## correlation and mix param are less than 1 constraint
    # theta[3] + gradient[3]*alpha < 1
    # gradient[3]*alpha < 1 - theta[3]
    # if gradient[2] > 0:
    #     alpha < (1 - theta[3])/gradient[3]
    # elif gradient[2] < 0:
    #     alpha < (theta[3] - 1)/gradient[3]
    for param_val, grad_val in zip(theta[2:], grad[2:]):
        if grad_val > 1e-6:
            max_alpha = min(max_alpha, (1-param_val)/grad_val)
        elif grad_val < -1e-6:
            max_alpha = min(max_alpha, (param_val-1)/grad_val)

    return max_alpha

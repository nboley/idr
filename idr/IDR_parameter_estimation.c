#include "IDR_parameter_estimation.h"

double RationalApproximation(double t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) 
        / (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

double NormalCDFInverse(double p)
{
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}

void calculate_quantiles(
    double rho,
    int n_samples,
    double* x_cdf, 
    double* y_cdf,
    double* updated_density
    )
{
    int i;
    for(i=0; i<n_samples; ++i)
    {
        double a = pow(NormalCDFInverse(x_cdf[i]), 2) 
                   + pow(NormalCDFInverse(y_cdf[i]), 2);
        double b = NormalCDFInverse(x_cdf[i]) * NormalCDFInverse(y_cdf[i]);
        updated_density[i] = exp( 
            -log(1 - pow(rho, 2)) / 2 
            - rho/(2 * (1 - pow(rho, 2))) * (rho*a-2*b)
            );
    }
}

double cost_function(
    double rho,
    int n_samples,
    double* x_cdf, 
    double* y_cdf,
    double* ez )
{

    double* new_density = (double*) calloc(n_samples, sizeof(double));
    calculate_quantiles(rho, 
                        n_samples,
                        x_cdf, 
                        y_cdf, 
                        new_density);

    double cop_den = 0.0;
    int i;
    for(i=0; i<n_samples; ++i)
    {
        cop_den = cop_den + (ez[i] * log(new_density[i]));
    }
    free(new_density);
    return -cop_den;
}

double maximum_likelihood(
    int n_samples,
    double* x_cdf,
    double* y_cdf,
    double* ez)
{
    double ax = -0.998;
    double bx = 0.998;
    double tol = 0.00001;

    /*  c is the squared inverse of the golden ratio */
    const double c = (3. - sqrt(5.)) * .5;

    /* Local variables */
    double a, b, d, e, p, q, r, u, v, w, x;
    double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

    /* eps is approximately the square root of the relative machine precision.*/
    eps = DBL_EPSILON;
    tol1 = eps + 1.;/* the smallest 1.000... > 1 */
    eps = sqrt(eps);

    a = ax;
    b = bx;
    v = a + c * (b - a);
    w = v;
    x = v;

    d = 0.;/* -Wall */
    e = 0.;
    fx = cost_function(x, n_samples, x_cdf, y_cdf, ez);
    fv = fx;
    fw = fx;
    tol3 = tol / 3.;

    for(;;)
    {
        xm = (a + b) * .5;
        tol1 = eps * fabs(x) + tol3;
        t2 = tol1 * 2.;

        /* check stopping criterion */
        if (fabs(x - xm) <= t2 - (b - a) * .5) break;
        p = 0.;
        q = 0.;
        r = 0.;
        if (fabs(e) > tol1)
        { /* fit parabola */
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.;
            if (q > 0.) p = -p; else q = -q;
            r = e;
            e = d;
        }

        if (fabs(p) >= fabs(q * .5 * r) ||
            p <= q * (a - x) || p >= q * (b - x))
        { /* a golden-section step */
            if (x < xm) e = b - x; else e = a - x;
            d = c * e;
        }
        else
        { /* a parabolic-interpolation step */
            d = p / q;
            u = x + d;
            /* f must not be evaluated too close to ax or bx */
            if (u - a < t2 || b - u < t2)
            {
                d = tol1;
                if (x >= xm) d = -d;
            }
        }

        /* f must not be evaluated too close to x */
        if (fabs(d) >= tol1)
            u = x + d;
        else if (d > 0.)
            u = x + tol1;
        else
            u = x - tol1;

        fu = cost_function(u, n_samples, x_cdf, y_cdf, ez);

        /*  update  a, b, v, w, and x */
        if (fu <= fx)
        {
            if (u < x) b = x; else a = x;
            v = w;    w = x;   x = u;
            fv = fw; fw = fx; fx = fu;
        }
        else
        {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x)
            {
                v = w; fv = fw;
                w = u; fw = fu;
            }
            else if (fu <= fv || v == x || v == w)
            {
                v = u; fv = fu;
            }
        }
    }
    return x;
}

double gaussian_loglikelihood(
    int n_samples,
    
    double*  x1_pdf, 
    double*  x2_pdf,
    double*  x1_cdf, 
    
    double*  y1_pdf, 
    double*  y2_pdf,
    double*  y1_cdf, 
    
    double p, double rho)
{
    double* density_c1 = (double*) calloc( sizeof(double), n_samples );
    double l0 = 0.0;

    calculate_quantiles(rho, n_samples, x1_cdf, y1_cdf, density_c1);
    int i;
    for(i=0; i<n_samples; ++i)
    {
        l0 = l0 + log(p*density_c1[i]*x1_pdf[i]*y1_pdf[i] 
                      + (1.0-p)*x2_pdf[i]*y2_pdf[i]);
    }
    free(density_c1);
    return l0;
}

void estep_gaussian(
    int n_samples,
    double* x1_pdf, double* x2_pdf,
    double* x1_cdf, double* x2_cdf, 
    double* y1_pdf, double* y2_pdf,
    double* y1_cdf, double* y2_cdf, 
    double* ez, double p, double rho)

{
    /* update density_c1 */
    double* density_c1 = (double*) calloc( sizeof(double), n_samples );
    calculate_quantiles(rho, n_samples, x1_cdf, y1_cdf, density_c1);

    /*
      In genreal for a mixture of two gaussians we would need to 
      do the following but, since the second gaussian has correlation
      0, then density_c2[i] is 1 for all i.
      
      double* density_c2 = (double*) calloc( sizeof(double), n_samples );
      calculate_quantiles(0, n_samples, x2_cdf, y2_cdf, density_c2);
    */
    int i;
    for(i=0; i<n_samples; ++i)
    {
        /* Technically this shoudl be  
           ... +(1-p)*(1-density_c1[i])*x2_pdf[i]*y2_pdf[i]) 
        but since hte noise component has correlation 0,
        density_c2[i] is always 1.0
        */
        double numerator = p * density_c1[i] * x1_pdf[i] * y1_pdf[i];
        double denominator = numerator + (1-p) * 1.0 * x2_pdf[i] * y2_pdf[i];
        ez[i] = numerator/denominator;
        assert( !isnan(ez[i]) );
    }
    free(density_c1);
    // we don't use this - see above
    // free(density_c2);
}

/* build the properly normalized cumsum array for bin_dens,
   store it into bin_cumsum, and return the sum */
double
build_cumsum(int nbins, double* bin_dens, double* bin_cumsum)
{
    /* normalize the bin counts */
    double cumsum = 0;
    int i;
    for(i=0; i<nbins; ++i)
    {
        cumsum += bin_dens[i];
        bin_cumsum[i] = cumsum;
    }
    
    double sum_ez = cumsum;
    cumsum = 0;
    for( i=0; i<nbins; i++ )
    {
        double prev_cumsum = cumsum;
        cumsum = bin_cumsum[i];
        bin_cumsum[i] = (cumsum + prev_cumsum)/(2.*sum_ez);
        assert( !isnan(bin_cumsum[i]));
        assert( 0 < bin_cumsum[i] &&  bin_cumsum[i] < 1);
    }

    return sum_ez;
}

/* use a histogram estimator to estimate the marginal distributions */
void estimate_marginals(
    int n_samples,
    int* input, 
    double* pdf_1, 
    double* pdf_2,
    double* cdf_1, 
    double* cdf_2, 

    /* the estimated mixture paramater for each point */
    double* ez, 
    
    int nbins)
{
    /* counter we will use throughout the script */
    int i;
            
    /* bin the observations */
    double* bin_dens_1 = (double*) calloc(sizeof(double), nbins);
    double* bin_cumsum_1 = (double*) calloc(sizeof(double), nbins);
    double* bin_dens_2 = (double*) calloc(sizeof(double), nbins);
    double* bin_cumsum_2 = (double*) calloc(sizeof(double), nbins);
    
    for(i=0; i<n_samples; ++i)
    {
        assert( input[i] >= 0 && input[i] < n_samples);
        int bin_i = (nbins*input[i])/n_samples;
        bin_dens_1[bin_i] += ez[i];
        bin_dens_2[bin_i] += (1-ez[i]);        
    }
    
    /* build the cumsum arrays, and return the total sum for each component */
    double sum_ez = build_cumsum(nbins, bin_dens_1, bin_cumsum_1);
    double sum_ez_comp = build_cumsum(nbins, bin_dens_2, bin_cumsum_2);
    
    /* normalize the bin densities */
    for(i=0; i<nbins; ++i)
    {
        bin_dens_1[i] = (bin_dens_1[i]+1)*nbins/(n_samples*(sum_ez+nbins));
        assert( !isnan(bin_dens_1[i]));
        bin_dens_2[i] = (bin_dens_2[i]+1)*nbins/(n_samples*(sum_ez_comp+nbins));
        assert( !isnan(bin_dens_2[i]));
    }

    /* set the pdf variables */
    for(i=0; i<n_samples; ++i)
    {
        int bin_i = (nbins*input[i])/n_samples;
        pdf_1[i] = bin_dens_1[bin_i];
        pdf_2[i] = bin_dens_2[bin_i];
        cdf_1[i] = bin_cumsum_1[bin_i];
        assert( 0 < cdf_1[i] &&  cdf_1[i] < 1);
        cdf_2[i] = bin_cumsum_2[bin_i];
        assert( 0 < cdf_2[i] &&  cdf_2[i] < 1);

    }

    free(bin_dens_1);
    free(bin_dens_2);
    free(bin_cumsum_1);
    free(bin_cumsum_2);
}

void mstep_gaussian(
    double* p0, double* rho,
    int n_samples,
    double* x1_cdf, 
    double* y1_cdf,
    double* ez)
{
    *rho = maximum_likelihood(
        n_samples, x1_cdf, y1_cdf, ez);

    double sum_ez = 0;
    int i;
    for(i = 0; i < n_samples; i++)
    { 
        sum_ez += ez[i]; 
    }
    *p0 = (sum_ez+1)/((double)n_samples+1);
}

struct OptimizationRV
em_gaussian(
    int n_samples,
    int* x, 
    int* y,
    double* IDRs,
    int print_status_msgs )
{
    int i;
    
    double* ez = (double*) malloc( sizeof(double)*n_samples );
    for(i = 0; i<n_samples; i++)
    {
        if(x[i] < n_samples/2) {
            ez[i] = 0.5;
        } else {
            ez[i] = 0.5;
        }
    }

    /* initialize the default configuration options */
    double p0 = -1;
    double rho = -1;
    double eps = 1e-2;

    /* Initialize the set of break points for the histogram averaging */
    int n_bins = 50;
    if( n_samples/5-2 < n_bins )
        n_bins = n_samples/5-2;
    /*
     * CDF and PDF vectors for the input vectors.
     * Updated everytime for a EM iteration.
     */
    double* x1_pdf = (double*) calloc(sizeof(double), n_samples);
    double* x2_pdf = (double*) calloc(sizeof(double), n_samples);
    double* x1_cdf = (double*) calloc(sizeof(double), n_samples);
    double* x2_cdf = (double*) calloc(sizeof(double), n_samples);
    
    double* y1_pdf = (double*) calloc(sizeof(double), n_samples);
    double* y2_pdf = (double*) calloc(sizeof(double), n_samples);
    double* y1_cdf = (double*) calloc(sizeof(double), n_samples);
    double* y2_cdf = (double*) calloc(sizeof(double), n_samples);
    
    /* Likelihood vector */
    double likelihood[3] = {0,0,0};
    double prev_p = -1;
    double prev_rho = -1;
    
    int iter_counter;
    for(iter_counter=0;;iter_counter++)
    {
        estimate_marginals(n_samples, x, 
                           x1_pdf, x2_pdf, 
                           x1_cdf, x2_cdf,
                           ez,
                           n_bins );
    
        estimate_marginals(n_samples, y, 
                           y1_pdf, y2_pdf, 
                           y1_cdf, y2_cdf,
                           ez,
                           n_bins );
        
        mstep_gaussian(&p0, &rho, n_samples, 
                       x1_cdf, y1_cdf, ez);

        estep_gaussian(n_samples,
                       x1_pdf, x2_pdf, 
                       x1_cdf, x2_cdf,
                       y1_pdf, y2_pdf, 
                       y1_cdf, y2_cdf,
                       ez, 
                       p0, rho);
        
        double l = gaussian_loglikelihood(
            n_samples,
            x1_pdf, x2_pdf, x1_cdf, 
            y1_pdf, y2_pdf, y1_cdf, 
            p0, rho);

        /* update the likelihood list */
        likelihood[0] = likelihood[1];
        likelihood[1] = likelihood[2];
        likelihood[2] = l;
        
        /* print out the likelihood after each iteration */
        if(print_status_msgs) {
            fprintf(stderr, "%i\t%e\t%e\t%e\n", iter_counter, p0, rho, l);
        }
        
        if (iter_counter > 3)
        {
            /* Aitken acceleration criterion checking for breaking the loop */
            double a_cri = likelihood[0] + (
                likelihood[1]-likelihood[0])
                / (1-(likelihood[2]-likelihood[1])/(
                       likelihood[1]-likelihood[0]));
            if ( fabs(a_cri-likelihood[2]) <= eps 
                 || likelihood[2] < likelihood[1] )
                 //&& fabs(p0 - prev_p) <= eps
                 //&& fabs(rho - prev_rho) <= eps)
            { break; }
        }
        
        prev_rho = rho;
        prev_p = p0;

    }
    
    for(i=0; i<n_samples; ++i)
    {
        IDRs[i] = 1.0 - ez[i];
    }
    
    free(ez);
    free(x1_pdf);
    free(x2_pdf);
    free(x1_cdf);
    free(x2_cdf);

    free(y1_pdf);
    free(y2_pdf);
    free(y1_cdf);
    free(y2_cdf);

    struct OptimizationRV rv = {iter_counter-1, rho, p0};
    return rv;
}

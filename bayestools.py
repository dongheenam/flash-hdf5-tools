
"""
DEPENDENCIES
"""
import sys

import emcee
import multiprocessing
import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect
from scipy.optimize import root_scalar
from scipy.special import erf
from scipy.stats import binom

"""
CONSTANTS
"""
M_MIN = 5e-7
M_MAX = 2e-2 # 2e-2
M_T_UPPER = 1.5e-3

# MCMC parameters
N_WALKERS = 96
N_BURNIN = 100
N_CHAIN = 500

""" THE CHABRIER IMF """
# the lognormal part
def imf_ln(theta, x_elem) :
    sigma, mu, m_t, gamma = theta
    return 1/(x_elem*sigma*np.sqrt(2*np.pi))*np.exp(-0.5*( (np.log10(x_elem)-np.log10(mu))/sigma )**2)

# the powerlaw part
def imf_pl(theta, x_elem) :
    sigma, mu, m_t, gamma = theta
    return x_elem**(gamma-1)

# the full Chabrier IMF
def imf_chab(theta, x_elem, pl_factor=1.0, A=1.0) :
    sigma, mu, m_t, gamma = theta
    if x_elem < m_t :
        return A*imf_ln(theta, x_elem)
    else :
        return A*pl_factor*imf_pl(theta, x_elem)

# normalise the IMF and evaluate at every x
def gen_imf_chab(theta, x) :
    sigma, mu, m_t, gamma = theta

    # first find the coefficient to make the imf continuous
    cont = root_scalar(lambda x: imf_ln(theta, m_t) - x*imf_pl(theta, m_t), x0=1.0, x1=0.9)
    pl_factor = cont.root
    #pl_factor = 0.4 * np.exp(-9.43e-2 * ( (np.log(m_t)-np.log(mu))/sigma )**2 ) * m_t**(-gamma) / sigma

    # then normalise the imf
    norm_factor1, _ = quad(lambda x: imf_ln(theta, x), M_MIN, m_t)
    norm_factor2, _ = quad(lambda x: pl_factor*imf_pl(theta, x), m_t, M_MAX)
    #norm_factor1  = -2.88586*sigma*erf( 0.307*(np.log(M_MIN)-np.log(mu)) )
    #norm_factor1 +=  2.88586*sigma*erf( 0.307*(np.log(m_t)-np.log(mu)) )
    #norm_factor1 /= np.sqrt(2*np.pi) * sigma
    #norm_factor2 = pl_factor/gamma * (M_MAX**gamma - m_t**gamma)
    try :
        A = float(1/(norm_factor1 + norm_factor2))
    except ZeroDivisionError :
        sys.exit(f"{norm_factor1}, {norm_factor2}")

    # return IMF(x)
    return np.array([imf_chab(theta, x_elem, pl_factor=pl_factor, A=A) for x_elem in x])

""" model and probility distributions """
# define the prior
def ln_prior(theta) :
    sigma, mu, m_t, gamma = theta

    in_range = (sigma>0 and M_MIN<mu<m_t and M_MIN<m_t<M_MAX and -4<gamma<0)
    if in_range :
        # prior goes here
        #return (1+b**2)**(-1.5)
        return 0
    else :
        # out of range
        return -np.inf

# define the likelihood
def ln_likelihood(theta, x) :
    imfs = gen_imf_chab(theta, x)
    if len(imfs[imfs==0]) == 0 :
        ll = np.sum(np.log(imfs))
        return ll
    else :
        return -np.inf

# the probability
def ln_probability(theta, x) :
    lp = ln_prior(theta)
    if not np.isfinite(lp) :
        return -np.inf
    else :
        return lp + ln_likelihood(theta, x)

"""
MACRO FUNCTIONS
"""
def fit_imf(masses_obs) :
    """
    runs MCMC to fit the sink masses to the Chabrier IMF
    """

    """ initialisation """
    print("initialising emcee...")
    # initial guess
    theta_names = ['sigma', 'mu', 'm_t', 'gamma']
    theta_guess = [0.5, 5e-4, 1e-3, -1.35]
    n_theta = len(theta_names)
    theta_0 = theta_guess + theta_guess*(0.1*np.random.random(size=(N_WALKERS,n_theta)))

    with multiprocessing.Pool() as pool :
        # initiate the sampler
        sampler = emcee.EnsembleSampler(N_WALKERS, n_theta, ln_probability, args=[masses_obs], pool=pool)

        # burnin runs
        print("running burnins...")
        pos, prob, state = sampler.run_mcmc(theta_0, N_BURNIN, progress=True)

        #Some helper code to help eliminate the worst chains:
        best_chain = np.argmax(prob)
        poor_chains = np.where(prob < np.percentile(prob, 33))
        for ix in poor_chains:
            pos[ix] = pos[best_chain]

        # the production run
        print("running productions...")
        sampler.reset()
        pos_final, _, _ = sampler.run_mcmc(pos, N_CHAIN, progress=True)

    # print out the result
    print("fininshed!")
    results = np.zeros((n_theta,3))
    for i in range(n_theta):
        quarts = np.percentile(pos_final[:, i], [16, 50, 84])
        ranges = np.diff(quarts)
        results[i] = (quarts[1], ranges[1], ranges[0])
        print(f"{theta_names[i]:10s}: {results[i,0]:.3E} (+{results[i,1]:.3E} -{results[i,2]:.3E})")

    return results

def hist_errors(hist, poisson=False) :
    """
    use Bayes theorem to calculate the error bars for a histogram
    """
    # total number
    N = np.sum(hist)
    # result
    errors = np.zeros((2, len(hist)))
    medians = np.zeros(len(hist))

    # Calculate poisson error?
    if poisson :
        for i in range(len(hist)) :
            p = hist[i]/N
            error_poisson = np.sqrt(p*(1-p)/N)*N

            medians[i] = hist[i]
            errors[:, i] = [error_poisson]*2

    # use Bayes theorem instead
    else :
        # the posterior function is just ( (N+1) * P(X=k), X~Binom(N,f)) )
        def post(f) :
            posterior = (N+1) * binom(N, f).pmf(k)
            return posterior

        # find x such that \int_0^x func(x') dx' = target
        def find_cum_at(func, target) :
            cumulative = lambda x: quad(func, 0, x)[0] - target
            try :
                result = bisect(cumulative, 0.0, 1.0)
            except ValueError :
                sys.exit(f"k={k}, f(a)={cumulative(0.0)}, f(b)={cumulative(1.0)}")
            return result

        # iterate over each bin
        for i in range(len(hist)) :
            print(f"calculating errors: {i/len(hist)*100:.2f}%", end='\r')
            k = hist[i]

            # only perform Bayesian error for small bins
            if k < np.sqrt(N)/2 :
                # find the 16th percentile for the posterior function
                f_16 = find_cum_at(post, 0.16)

                # find the 84th percentile for the posterior function
                f_84 = find_cum_at(post, 0.84)

                medians[i] = N*find_cum_at(post, 0.50)
                errors[0, i] = np.abs(medians[i]-N*find_cum_at(post, 0.16))
                errors[1, i] = np.abs(medians[i]-N*find_cum_at(post, 0.84))

            # for the big bins, calculate Poisson error
            else :
                p = hist[i]/N
                error_poisson = np.sqrt(p*(1-p)/N)*N

                medians[i] = hist[i]
                errors[:, i] = [error_poisson]*2

    print("error calculation complete!")
    return medians, errors

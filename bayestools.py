
"""
DEPENDENCIES
"""
import emcee
import multiprocessing
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import erf

"""
CONSTANTS
"""
M_MIN = 1e-5
M_MAX = 20.0

# MCMC parameters
N_WALKERS = 48
N_BURNIN = 100
N_CHAIN = 500

""" THE CHABRIER IMF """
# the lognormal part
def imf_ln(theta, x_elem) :
    sigma, mu, m_t, gamma, m_max = theta
    return 1/(x_elem*sigma*np.sqrt(2*np.pi))*np.exp(-0.5*( (np.log10(x_elem)-np.log10(mu))/sigma )**2)

# the powerlaw part
def imf_pl(theta, x_elem) :
    sigma, mu, m_t, gamma, m_max = theta
    return x_elem**(gamma-1)

# the full Chabrier IMF
def imf_chab(theta, x_elem, pl_factor=1.0, A=1.0) :
    sigma, mu, m_t, gamma, m_max = theta
    if x_elem < m_t :
        return A*imf_ln(theta, x_elem)
    elif x_elem < m_max :
        return A*pl_factor*imf_pl(theta, x_elem)
    else :
        return 0

# normalise the IMF and evaluate at every x
def gen_imf_chab(theta, x) :
    sigma, mu, m_t, gamma, m_max = theta

    # first find the coefficient to make the imf continuous
    cont = root_scalar(lambda x: imf_ln(theta, m_t) - x*imf_pl(theta, m_t), x0=1.0, x1=0.9)
    pl_factor = cont.root

    # then normalise the imf
    norm_factor1, _ = quad(lambda x: imf_ln(theta, x), M_MIN, m_t)
    norm_factor2, _ = quad(lambda x: pl_factor*imf_pl(theta, x), m_t, m_max)
    A = 1/(norm_factor1 + norm_factor2)

    # return IMF(x)
    return np.array([imf_chab(theta, x_elem, pl_factor=pl_factor, A=A) for x_elem in x])

"""
MACRO FUNCTIONS
"""
def fit_imf(masses_obs) :
    """
    runs MCMC to fit the cdf to the Chabrier IMF
    """

    """ model and probility distributions """
    # define the prior
    def ln_prior(theta) :
        sigma, mu, m_t, gamma, m_max = theta

        in_range = (sigma>0 and M_MIN<mu<m_t and M_MIN<m_t<m_max and -3<gamma<0)
        if in_range :
            # prior goes here
            #return (1+b**2)**(-1.5)
            return np.log(1/m_max)
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

    """ initialisation """
    print("initialising emcee...")
    # initial guess
    theta_names = ['sigma', 'mu', 'm_t', 'gamma', 'm_max']
    theta_guess = [0.3, 0.2, 0.5, -1.35, 30.0]
    n_theta = len(theta_names)
    theta_0 = theta_guess + 1e-1*np.random.random(size=(N_WALKERS,n_theta))

    with multiprocessing.Pool() as pool :
        # initiate the sampler
        sampler = emcee.EnsembleSampler(N_WALKERS, n_theta, ln_probability, args=[masses_obs])

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


def fit_cdf_chab(x_obs, y_obs, y_err) :
    """
    runs MCMC to fit the cdf to the Chabrier IMF
    """

    """ model and probility distributions """
    # define the prior
    def ln_prior(theta) :
        A, sigma, mu, m_t, k2, gamma, ln_f = theta

        in_range = (A>0 and sigma>0 and 0.1<m_t<2.0 and M_MIN<mu<M_MAX and k2>0 and -3<gamma<0 and -10<ln_f<3)
        if in_range :
            # prior goes here
            return 0
        else :
            # out of range
            return -np.inf

    # define the likelihood
    def ln_likelihood(theta, x, y, y_err) :
        ln_f = theta[-1]

        model = np.array([cdf_chab(theta, x_elem) for x_elem in x])

        sigma2 = y_err**2 + model**2 * np.exp(2*ln_f)
        lhd = -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))
        return lhd

    # the probability
    def ln_probability(theta, x, y, y_err) :
        lp = ln_prior(theta)
        if not np.isfinite(lp) :
            return -np.inf
        else :
            return lp + ln_likelihood(theta, x, y, y_err)

    """ initialisation """
    print("initialising emcee...")
    # initial guess
    # A, sigma, mu, m_t, k2, gamma, ln_f = theta
    theta_0 = [1.0, 0.5, 0.5, 1.0, 0.5, -1.35, -2.0] + 1e-3*np.random.random(size=(N_WALKERS,7))

    with multiprocessing.Pool() as pool :
        # initiate the sampler
        sampler = emcee.EnsembleSampler(N_WALKERS, 7, ln_probability, args=[x_obs, y_obs, y_err])

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
    names = ['A', 'sigma', 'mu', 'm_t', 'k2', 'gamma', 'ln_f']
    results = np.zeros((len(names),3))
    for i in range(len(names)):
        quarts = np.percentile(pos_final[:, i], [16, 50, 84])
        ranges = np.diff(quarts)
        results[i] = (quarts[1], ranges[1], ranges[0])
        print(f"{names[i]:10s}: {results[i,0]:.3E} (+{results[i,1]:.3E} -{results[i,2]:.3E})")

    return results

def fit_cdf(x_obs, y_obs, y_err) :
    """
    runs MCMC to fit the cdf to the model CDF = 1 - a*M**b
    which corresponds to IMF = k M**(-x-1) dM as
    k = a*b, x = -b
    """

    """ model and probility distributions """
    # define the prior
    def ln_prior(theta) :
        a, b, ln_f = theta

        a_in_range = a>0
        b_in_range = True
        f_in_range = -10<ln_f<2
        if (a_in_range and b_in_range and f_in_range) :
            # prior goes here
            return 0
        else :
            # out of range
            return -np.inf

    # define the likelihood
    def ln_likelihood(theta, x, y, y_err) :
        a, b, ln_f = theta
        model = 1 - a*x**b

        sigma2 = y_err**2 + model**2 * np.exp(2*ln_f)
        return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

    # the probability
    def ln_probability(theta, x, y, y_err) :
        lp = ln_prior(theta)
        if not np.isfinite(lp) :
            return -np.inf
        else :
            return lp + ln_likelihood(theta, x, y, y_err)

    """ initialisation """
    print("initialising emcee...")
    # initial guesslog_f
    theta_0 = [1.0, -1.35, -2.0] + 1e0*np.random.random(size=(N_WALKERS,3))

    with multiprocessing.Pool() as pool :
        # initiate the sampler
        sampler = emcee.EnsembleSampler(N_WALKERS, 3, ln_probability, args=[x_obs, y_obs, y_err])

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
    names = ['a','b', 'ln_f']
    results = np.zeros((3,3))
    for i in range(3):
        quarts = np.percentile(pos_final[:, i], [16, 50, 84])
        ranges = np.diff(quarts)
        results[i] = (quarts[1], ranges[1], ranges[0])
        print(f"{names[i]:10s}: {results[i,0]:.3E} (+{results[i,1]:.3E} -{results[i,2]:.3E})")

    return results

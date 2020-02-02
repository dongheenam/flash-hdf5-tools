
"""
DEPENDENCIES
"""
import emcee
import numpy as np

"""
CONSTANTS
"""

# initial guesses
B_RANGE = (-3,2)
A_0 = 1.0
B_0 = -1.35


# MCMC parameters
N_WALKERS = 1000
N_BURNIN = 100
N_CHAIN = 5000


"""
MACRO FUNCTIONS
"""
def fit_cdf(x_obs, y_obs, y_err) :
    """
    runs MCMC to fit the cdf to the model CDF = 1 - a*M**b
    which corresponds to IMF = k M**(-x-1) dM as
    k = a*b, x = -b
    """

    """ probility distributions """
    # define the prior
    def ln_prior(theta) :
        a, b, ln_f = theta

        a_in_range = a>0
        b_in_range = (b>B_RANGE[0] and b<B_RANGE[1])
        if (a_in_range and b_in_range) :
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
        if not np.isfinite(lp):
            return -np.inf
        else :
            return lp + ln_likelihood(theta, x, y, y_err)

    """ initialisation """
    print("initialising emcee...")
    # initial guesslog_f
    theta_0 = [A_0,B_0, -2.0] + 1e0*np.random.random(size=(N_WALKERS,3))

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

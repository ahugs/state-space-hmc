from densities.base_density import BaseDensity
from mcmc.metropolis_hastings_mcmc import MetropolisHastingsMCMC
import numpy as np
import numpy.linalg as npla

from scipy.stats import multivariate_normal


class MultivariateNormal(BaseDensity):

    def __init__(self, mu, C):
        super().__init__()
        self.mu = mu
        self.C = C

    def evaluate(self, x):
        ''' Evaluate the log density at x'''
        return multivariate_normal.logpdf(x, mean=self.mu, cov=self.C)

    def grad_evaluate(self, x):
        ''' Evaluate the gradient of the log density at x'''
        return -1 * np.dot(npla.inv(self.C), (x - self.mu))

    def sample(self):
        return multivariate_normal.rvs(mean=self.mu, cov=self.C, size=1)



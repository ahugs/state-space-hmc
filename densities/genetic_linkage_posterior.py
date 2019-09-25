from densities.base_density import BaseDensity
from mcmc.metropolis_hastings_mcmc import MetropolisHastingsMCMC
from densities.multivariate_normal_density import MultivariateNormal
import numpy as np


class GeneticLinkagePosterior(BaseDensity):

    def __init__(self, y):
        super().__init__()
        self.y = y

    def evaluate(self, x):
        if x > 1 or x < 0:
            return -1 * np.infty
        else:
            return self.y[0] * np.log((2 + x) / 4.) + (self.y[1] + self.y[2]) * np.log((1 - x) / 4.) \
                   + self.y[3] * np.log(x / 4.)

    def grad_evaluate(self, x):
        print('x:{}'.format(x))
        if x > 1 or x < 0:
            return -1 * np.infty
        else:
            return -self.y[0]/((2 + x)**2) - (self.y[1] + self.y[2])/((1 - x)**2) - self.y[3]/(x**2)

    def sample(self):
        mcmc = MetropolisHastingsMCMC(target_density=self, proposal_density=MultivariateNormal(mu=0, C=0.1),
                                      q_0=np.random.beta(1, 1), n=10000)
        return mcmc.run_chain()[0, -1]

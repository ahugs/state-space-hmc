from mcmc.base_mcmc import BaseMCMC

import numpy as np
from numpy.random import rand


class MetropolisHastingsMCMC(BaseMCMC):

    def __init__(self, target_density, proposal_density, q_0, n=10000, n_chains=4):
        super().__init__(q_0=q_0, n=n, n_chains=n_chains)
        self.target_density = target_density
        self.proposal_density = proposal_density

    def propose(self, current):
        return current, current + self.proposal_density.sample()

    def accept_reject(self, current, proposal):
        p = np.min([np.exp(self.target_density.evaluate(proposal) - self.target_density.evaluate(current)), 1])
        if rand() < p:
            return proposal
        else:
            return current

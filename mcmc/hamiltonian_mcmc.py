import numpy as np

from densities.multivariate_normal_density import MultivariateNormal
from integrators.leapfrog_integrator import LeapfrogIntegrator
from mcmc.base_mcmc import BaseMCMC


class HamiltonianMCMC(BaseMCMC):

    def __init__(self, target_density, q_0, proposal_density=None, n=10000, n_chains=4):
        super().__init__(q_0=q_0, n=n, n_chains=n_chains)
        self.target_density = target_density
        if proposal_density is not None:
            self.proposal_density = proposal_density
        else:
            self.proposal_density = MultivariateNormal(np.zeros(len(q_0)), 1*np.eye(len(q_0)))

    def propose(self, current):
        # Make a Metropolis Hastings Propsoal
        p_curr = self.proposal_density.sample()
        leapfrog_integrator = LeapfrogIntegrator(self.target_density, self.proposal_density, current, p_curr)
        q, p = leapfrog_integrator.integrate()
        return (current, p_curr), (q, -1 * p)

    def accept_reject(self, current, proposal):
        # Accept/reject proposal
        q, p = proposal
        q_curr, p_curr = current

        acceptance_prob = np.min([1, np.exp(self.target_density.evaluate(q) - self.target_density.evaluate(q_curr) \
                                 + self.proposal_density.evaluate(p) - self.proposal_density.evaluate(p_curr))])
        if np.random.rand() < acceptance_prob:
            return q
        else:
            return q_curr








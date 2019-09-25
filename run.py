from mcmc.metropolis_hastings_mcmc import MetropolisHastingsMCMC
from mcmc.hamiltonian_mcmc import HamiltonianMCMC
from densities.genetic_linkage_posterior import GeneticLinkagePosterior
from densities.multivariate_normal_density import MultivariateNormal
from numpy.random import normal
import numpy as np


target_density = GeneticLinkagePosterior(y=[125, 18, 20, 34])
target_density = MultivariateNormal(0, 1*np.eye(1))
# proposal_density = MultivariateNormal(mu=0, C=0.1)
# print(target_density.sample())
mh_mcmc = HamiltonianMCMC(target_density, q_0=np.array([0]), n=10000)
chain = mh_mcmc.run_chain()
print(np.mean(chain[0,2000:10000]))
print(chain[:,9000:10000])

#print(target_density.evaluate(23))
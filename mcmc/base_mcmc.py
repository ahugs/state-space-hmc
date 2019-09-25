from abc import ABC, abstractmethod
import numpy as np


class BaseMCMC(ABC):

    def __init__(self, q_0, n_chains, n):
        self.n = n
        #self.n_chains = n_chains
        self.chain = np.zeros(shape=(len(q_0), self.n))
        self.q_0 = q_0

    def run_chain(self):
        # self.chains[0,:] = self.q_0
        self.chain[:, 0] = self.q_0
        # for i in range(0, self.n_chains):
        #     for j in range(1, self.n):
        #         current, proposal = self.propose(self.chains[i, j - 1])
        #         if self.accept_reject(current, proposal):
        #             self.chains[i, j] = proposal
        #         else:
        #             self.chains[i, j] = self.chains[i, j-1]
        for i in range(1, self.n):
            current, proposal = self.propose(self.chain[:, i - 1])
            self.chain[:, i] = self.accept_reject(current, proposal)

        return self.chain

    @abstractmethod
    def propose(self, current):
        ''' Makes a proposal for next step in chain'''
        pass

    @abstractmethod
    def accept_reject(self, current, proposal):
        ''' Accepts/rejects proposal '''
        pass

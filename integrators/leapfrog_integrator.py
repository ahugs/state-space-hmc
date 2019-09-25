import numpy as np


class LeapfrogIntegrator(object):

    def __init__(self, target, proposal, q_0, p_0):
        self.target = target
        self.proposal = proposal
        self.q_0 = q_0
        self.p_0 = p_0
        self.D = len(q_0)

    def integrate(self, T=10, epsilon=0.0001):
        p = np.zeros(shape=(self.D, T))
        q = np.zeros(shape=(self.D, T))
        p[:, 0] = self.p_0
        q[:, 0] = self.q_0

        for t in range(1, T):
            p[:, t], q[:, t] = self._leapfrog_step(q[:, t - 1], p[:, t - 1], epsilon)

        return q[:, T - 1], p[:, T - 1]

    def _leapfrog_step(self, q_t, p_t, epsilon):
        p_t_half_e = p_t - epsilon/2. * (-1*self.target.grad_evaluate(q_t))
        q_t_e = q_t + epsilon * (-1*self.proposal.grad_evaluate(p_t_half_e))
        p_t_e = p_t_half_e - epsilon/2. * (-1*self.target.grad_evaluate(q_t_e))
        return p_t_e, q_t_e



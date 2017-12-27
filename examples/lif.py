import autograd.numpy as np
from autograd.scipy.special import expit

"""
Example custom likelihood. Leaky integrate and fire model of a neuron.
"""
class LIFLike():

    def __init__(self, g = .03, v_rest = 0., v_thresh = 15.,
                 phi = 0.025, t_max = 51, l = 1., ts = None):

        self.g = g
        self.phi = phi
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        self.t_max = t_max
        self.l = l
        self.ts = ts
        self.constant = self.calc_constant()
        self.const_mat = self.calc_const_mat()
        self.t_idx = np.arange(len(self.ts)), self.ts

    def log_like(self, s, t):

        v = np.multiply(np.exp(s),self.const_mat.T).T
        p = expit(v - self.v_thresh)
        logp = np.sum(np.log(1 - p), 1)
        logp = logp + np.multiply(t < self.t_max,
                                  -np.log(1 - p[self.t_idx]) +
                                  np.log(p[self.t_idx]))
        return np.nan_to_num(logp)

    def calc_constant(self):

        c_i = 0.
        constants = [c_i]

        for i in range(1, self.t_max+1):

            c_i = c_i + np.exp(-self.g*i)
            constants.append(c_i)

        return self.l*self.phi*np.array(constants)

    def calc_const_mat(self):

        if self.ts is None:
            return

        const_mat = np.zeros((len(self.ts), self.t_max + 1))

        for i, t in enumerate(self.ts):
            const_mat[i, :t+1] = self.constant[:t+1]

        return const_mat

class LIFSim():

    def __init__(self, g = .03, v_rest = 0., v_thresh = 15.,
                 phi = 0.025, t_max = 50, l = 1.):
        self.g = g
        self.phi = phi
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        self.t_max = t_max
        self.l = l
        self.constant = self.calc_constant()

    def sim(self, s):

        spikes = np.ones(len(s))*self.t_max + 1

        for u in range(1, self.t_max):

            v_i = np.exp(s) * self.constant[u]
            lambda_u = expit(v_i-self.v_thresh)
            spikes = np.where((np.random.binomial(1, lambda_u)) & (spikes > self.t_max),
                               u, spikes)

        return spikes

    def calc_constant(self):

        c_i = 0.
        constants = [c_i]

        for i in range(1, self.t_max):
            c_i = c_i + np.exp(-self.g * i)
            constants.append(c_i)

        return self.l * self.phi * np.array(constants)

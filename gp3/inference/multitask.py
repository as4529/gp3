import numpy as np
from scipy.linalg import toeplitz
from copy import copy
from gp3.kernels import TaskEmbed
from . import Vanilla
from gp3.utils.transforms import unit_norm

class Multitask(Vanilla):

    def __init__(self,
                 X,
                 y,
                 kernels,
                 mu=None,
                 obs_idx=None,
                 task_idx=None,
                 noise=1e-2,
                 q_dim=2,
                 q_var=1.):
        """

        Args:
            X (np.array): data
            y (np.array): output
            kernel (): kernel function to use for inference
            obs_idx (np.array): Indices of observed points (partial grid)
            noise (float): observation noise
        """

        super(Multitask, self).__init__(X, y, kernels, mu=mu, obs_idx=obs_idx,
                                        noise=noise, opt_idx=[0])
        self.q_dim = q_dim
        self.q_var = q_var
        self.task_idx = task_idx
        self.n_tasks = len(np.unique(task_idx))
        self.n = len(obs_idx) if obs_idx is not None else len(y)
        self.m = X.shape[0] * self.n_tasks
        self.mu = np.zeros(self.m) if mu is None else mu
        self.d = X.shape[1] + 1

        self.Q = np.random.normal(0, 1, size=(self.n_tasks,
                                        self.q_dim))*0.1
        self.kernels.insert(0, TaskEmbed(self.Q,
                                         self.n_tasks,
                                         self.q_dim))
        self.mvp = self.kron_mvp_mt
        self.X_dims.insert(0, [None])
        self.construct_Ks()

    def kron_mvp_mt(self, Ks, v):

        m = [k.shape[0] for k in Ks]
        n = m
        b = copy(v)

        for i in range(len(Ks)):
            a = np.reshape(b, (np.prod(m[:i], dtype=int),
                               n[i],
                               np.prod(n[i + 1:], dtype=int)))
            tmp = np.reshape(np.swapaxes(a, 2, 1),
                             (-1, n[i]))
            if i > 0:
                tmp = tmp.dot(Ks[i].T)
            else:
                tmp = tmp.dot(Ks[i]).dot(Ks[i].T)
            b = np.swapaxes(np.reshape(tmp,
                                       (a.shape[0],
                                        a.shape[2],
                                        m[i])), 2, 1)
        return np.reshape(b, np.prod(m, dtype=int))

    def grad_marginal_k(self, i, d):

        n_params = len(self.kernels[d].params)
        grads = np.zeros(n_params)
        grad_K = np.squeeze(self.kernel_grads[i][0](self.kernels[d].params,
                                                 self.X_dims[d][0],
                                                 self.X_dims[d]))
        self.grad_K = grad_K
        for j in range(n_params):
            if d > 0:
                K_grad_params = toeplitz(grad_K[:, j])
            else:
                K_grad_params = grad_K[:, :, j]
            Ks_grads = copy(self.Ks)
            Ks_grads[d] = K_grad_params
            grad_j = - 0.5 * self.stochastic_trace(Ks_grads) + \
                       0.5 * np.dot(self.alpha, self.K_prod(Ks_grads, self.alpha))
            grads[j] = grad_j
        return grads

    def loss_check(self, losses):
        """
        Checks conditions for loss decreasing

        Returns: True if condition satisfied

        """
        if len(losses) < 20:
            return False
        if sum(x <= y for x, y in zip(losses[-20:], losses[-10:])) > 10 and\
            losses[-1] - losses[-20] < 1e-3*abs(losses[-10]):
            return True

import numpy as np
from gp3.utils.optimizers import CG
from gp3.utils.structure import kron_list, kron_mvp, kron_list_diag
from .base import InfBase


"""
Class for Kronecker inference of GPs with Gaussian likelihood. Inspiration from GPML.

For references, see:

Wilson et al (2014),
Thoughts on Massively Scalable Gaussian Processes

Most of the notation follows R and W chapter 2

"""


class Vanilla(InfBase):

    def __init__(self, X, y, kernel, mu = None,
                 obs_idx=None, noise = 1e-6):
        """

        Args:
            X (np.array): data
            y (np.array): output
            kernel (): kernel function to use for inference
            obs_idx (np.array): Indices of observed points (partial grid)
            noise (float): observation noise
        """

        super(Vanilla, self).__init__(X, y, kernel,
                                      mu, obs_idx, noise=noise)
        self.opt = CG(self.cg_prod)
        self.root_eigdecomp = self.sqrt_eig()
        if obs_idx is not None:
            self.m = len(obs_idx)
        else:
            self.m = self.n

    def sqrt_eig(self):
        """
        Calculates square root of kernel matrix using
         fast kronecker eigendecomp.
        This is used in stochastic approximations
         of the predictive variance.

        Returns: Square root of kernel matrix

        """
        res = []

        for e, v in self.K_eigs:
            e_root_diag = np.sqrt(e)
            e_root = np.diag(e_root_diag)
            res.append(np.dot(np.dot(v, e_root), np.transpose(v)))

        res = kron_list(res)
        self.root_eigdecomp = res

        return res

    def variance(self, n_s):
        """
        Stochastic approximator of predictive variance.
         Follows "Massively Scalable GPs"
        Args:
            n_s (int): Number of iterations to run stochastic approximation

        Returns: Approximate predictive variance at grid points

        """

        if self.root_eigdecomp is None:
            self.sqrt_eig()

        var = np.zeros([self.m])
        diag = kron_list_diag(self.Ks)

        for i in range(n_s):
            g_m = np.random.normal(size = self.n)
            g_n = np.random.normal(size = self.n)

            right_side = np.dot(self.root_eigdecomp, g_m) +\
                         np.sqrt(self.noise)*g_n
            r = self.opt.cg(self.Ks, right_side)
            var += np.square(kron_mvp(self.Ks, r))

        return np.clip(diag - var/n_s, 0, 1e12).flatten()

    def variance_slow(self, n_s):

        K = kron_list(self.Ks)
        A = kron_list(self.Ks) + np.diag(np.ones(self.n) * self.noise)
        A_inv = np.linalg.inv(A)
        A_inv_chol = np.linalg.cholesky(A_inv)
        var = np.zeros([self.m])
        vars = []

        for i in range(n_s):
            eps = np.random.normal(size = self.n)
            r = np.dot(A_inv_chol, eps)
            var += np.square(np.dot(K, r))
            if i % 10 == 0:
                var_t = np.clip(np.diag(K) - var/i, 0, 1e12)
                vars.append(var_t)

        return np.clip(np.diag(K) - var/n_s, 0, 1e12).flatten(), vars

    def variance_exact(self):

        K = kron_list(self.Ks)
        A = kron_list(self.Ks) + np.diag(np.ones(self.n) * self.noise)
        A_inv = np.linalg.inv(A)

        return np.squeeze(np.diag(K) -\
                          np.diag(np.dot(K, A_inv).dot(K)))

    def predict_mean(self):
        """
        Predicts mean at X points

        Returns: f_pred(X)

        """

        return kron_mvp(self.Ks, self.alpha)

    def solve(self):
       """
       Uses linear conjugate gradients to solve for (K + noise)^{-1}y
       Returns:

       """

       self.alpha = self.opt.cg(self.Ks, self.y)

       return

    def cg_prod(self, Ks, p):
        """

        Args:
            p (np.array): potential solution to linear system

        Returns: product Ap (left side of linear system)

        """

        if self.obs_idx is not None:
            Wp = np.zeros(self.m)
            Wp[self.obs_idx] = p
            kprod = kron_mvp(Ks, Wp)[self.obs_idx]
        else:
            kprod = kron_mvp(Ks, p)

        return self.noise*p + kprod

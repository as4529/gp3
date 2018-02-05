import numpy as np
from gp3.utils.optimizers import CG
from gp3.utils.structure import kron_list, kron_mvp
from scipy.linalg import toeplitz
import scipy
from .base import InfBase


"""
Class for Kronecker inference of GPs. Inspiration from GPML.

For references, see:

Wilson et al (2014),
Thoughts on Massively Scalable Gaussian Processes

Most of the notation follows R and W chapter 2

"""


class Vanilla(InfBase):

    def __init__(self, kernel, X, y, mu = None,
                 obs_idx=None, verbose=False, noise = 1e-6):
        """

        Args:
            kernel (kernels.Kernel): kernel function to use for inference
            likelihood (likelihoods.Likelihood): likelihood
            X (np.array): data
            y (np.array): output
            tau (float): Newton line search hyperparam
            obs_idx (np.array): Indices of observed points (partial grid)
            verbose (bool): verbose or not
        """

        super(Vanilla, self).__init__(X, y, kernel,
                                      mu, obs_idx, noise=noise)
        self.opt = CG(self.cg_prod)

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

        for i in range(n_s):
            g_m = np.random.normal(size = self.m)
            g_n = np.random.normal(size = self.n)

            right_side = np.dot(self.root_eigdecomp, g_m) +\
                         np.sqrt(self.kernel.lengthscale**self.d + self.noise)*g_n
            r = self.opt.cg(self.Ks, right_side)
            var += np.square(kron_mvp(self.Ks, r))

        return np.clip(self.kernel.eval(self.kernel.params,
            np.array([[0.]]), np.array([[0.]]))**self.d - var/n_s, 0, 1e12).flatten()

    def variance_slow(self, n_s):

        K = kron_list(self.Ks)
        A = kron_list(self.Ks) + np.diag(np.ones(self.n)*self.noise)
        A_inv = np.linalg.inv(A)
        var = np.zeros([self.m])

        for i in range(n_s):
            r = np.random.multivariate_normal(mean = np.zeros(self.n), cov = A_inv)
            var += np.square(np.dot(K, r))

        return np.clip(self.kernel.eval(self.kernel.params,
            np.array([[0.]]), np.array([[0.]])) - var/n_s, 0, 1e12).flatten()


    def predict_mean(self):

        return kron_mvp(self.Ks, self.alpha)

    def solve(self):

       self.alpha = self.opt.cg(self.Ks, self.y)

       return

    def cg_prod(self, Ks, p):
        """

        Args:
            p (tfe.Variable): potential solution to linear system

        Returns: product Ap (left side of linear system)

        """

        if self.obs_idx is not None:
            Wp = np.zeros(self.m)
            Wp[self.obs_idx] = p
            kprod = kron_mvp(Ks, Wp)[self.obs_idx]
        else:
            kprod = kron_mvp(Ks, p)

        return self.noise*p + kprod

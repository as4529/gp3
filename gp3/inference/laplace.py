import sys
from tqdm import trange
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from gp3.utils.optimizers import CG
from gp3.utils.structure import kron_list, kron_mvp, kron_list_diag
from scipy.optimize import minimize
from .base import InfBase


"""
Class for Kronecker inference of GPs. Inspiration from GPML.

For references, see:

Flaxman and Wilson (2014),
Fast Kronecker Inference in Gaussian Processes with non-Gaussian Likelihoods

Rassmussen and Williams (2006),
Gaussian Processes for Machine Learning

Wilson et al (2012),
Fast Kernel Learning for Multidimensional Pattern Extrapolation

Wilson et al (2014),
Thoughts on Massively Scalable Gaussian Processes

Most of the notation follows R and W chapter 2, and Flaxman and Wilson

"""


class Laplace(InfBase):

    def __init__(self, X, y, kernel, likelihood, mu = None,
                 opt_kernel = False, tau=0.5, obs_idx=None, noise = 1e-6):
        """

        Args:
            X (np.array): data
            y (np.array): output
            kernel (): kernel function to use for inference
            likelihood (likelihoods.Likelihood): likelihood
            tau (float): Newton line search hyperparam
            obs_idx (np.array): Indices of observed points (partial grid)
            verbose (bool): verbose or not
        """

        super(Laplace, self).__init__(X, y, kernel, likelihood,
                                      mu, obs_idx, opt_kernel, noise=noise)

        self.alpha = np.zeros([X.shape[0]])
        self.W = np.zeros([X.shape[0]])
        self.grads = np.zeros([X.shape[0]])
        self.opt = CG(self.cg_prod)
        self.root_eigdecomp = None

        self.f = self.mu
        self.f_pred = self.f
        self.tau = tau

        self.grad_func = egrad(self.likelihood.log_like)
        self.hess_func = egrad(self.grad_func)

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
            e_root = np.diag(np.real(np.nan_to_num(e_root_diag)))
            res.append(np.real(np.dot(np.dot(v, e_root), np.transpose(v))))

        res = np.squeeze(kron_list(res))
        self.root_eigdecomp = res

        return res

    def run(self, max_it):
        """
        Runs Kronecker inference. Updates instance variables.

        Args:
            max_it (int): maximum number of iterations.

        Returns: max iterations, iteration number, objective

        """
        if self.obs_idx is not None:
            k_diag = np.ones(self.X.shape[0]) * 1e12
            k_diag[self.obs_idx] = self.noise
            self.k_diag = k_diag
            self.precondition = np.clip(1.0 / np.sqrt(self.k_diag),
                                                 0, 1)
        else:
            self.k_diag = None
            self.precondition = None

        delta = sys.float_info.max
        it = 0

        t = trange(max_it)

        for i in t:
            max_it, it, delta, step, psi = self.step(max_it, it, delta)
            t.set_description("Objective: " + '{0:.2f}'.format(psi) +
                              " | Step Size: " + '{0:.2f}'.format(step))
            if delta < 1e-9:
                break

        self.f_pred = kron_mvp(self.Ks, self.alpha) + self.mu
        self.update_derivs()

        return

    def step(self, max_it, it, delta):
        """
        Runs one step of Kronecker inference
        Args:
            max_it (int): maximum number of Kronecker iterations
            it (int): current iteration
            delta (np.array): change in step size

        Returns: max iteration, current iteration, previous objective,
         change in objective

        """

        self.f = kron_mvp(self.Ks, self.alpha) + self.mu
        if self.k_diag is not None:
            self.f += np.multiply(self.alpha, self.k_diag)
        psi = self.log_joint(self.f, self.alpha)
        self.update_derivs()

        b = np.multiply(self.W, self.f - self.mu) + self.grads
        if self.precondition is not None:
            z = self.opt.cg(self.Ks, np.multiply(self.precondition,
                            np.multiply(1.0/np.sqrt(self.W), b)))
        else:
            z = self.opt.cg(self.Ks, np.multiply(1.0/np.sqrt(self.W), b))

        delta_alpha = np.multiply(np.sqrt(self.W), z) - self.alpha
        step_size = self.line_search(delta_alpha, psi, 20)

        delta = step_size

        if delta > 1e-9:
            self.alpha = self.alpha + delta_alpha*step_size
            self.alpha = np.where(np.isnan(self.alpha),
                                  np.ones_like(self.alpha) * 1e-9, self.alpha)

        it = it + 1

        return max_it, it, delta, step_size, psi

    def line_search(self, delta_alpha, obj_prev, max_it, verbose = False):
        """
        Executes line search for optimal Newton step
        Args:
            delta_alpha (np.array): change in search direction
            obj_prev (np.array): previous objective value
            max_it (int): maximum number of iterations

        Returns: optimal step size

        """
        obj_search = sys.float_info.max
        min_obj = obj_prev
        step_size = 2.0
        opt_step = 0.0
        t = 1

        while t < max_it and obj_prev - obj_search < step_size * t:
            obj_prev, min_obj, delta_alpha, step_size, max_it, t, opt_step = \
                self.search_step(obj_prev, min_obj, delta_alpha, step_size,
                             max_it, t, opt_step)
        return opt_step


    def search_step(self, obj_prev, min_obj, delta_alpha,
                    step_size, max_it, t, opt_step):
        """
        Executes one step of a backtracking line search
        Args:
            obj_prev (np.array): previous objective
            obj_search (np.array): current objective
            min_obj (np.array): current minimum objective
            delta_alpha (np.array): change in step size
            step_size (np.array): current step size
            max_it (int): maximum number of line search iterations
            t (np.array): current line search iteration
            opt_step (np.array): optimal step size until now

        Returns: updated parameters
        """
        alpha_search = np.squeeze(self.alpha + step_size * delta_alpha)
        f_search = np.squeeze(kron_mvp(self.Ks, alpha_search)) + self.mu

        if self.k_diag is not None:
            f_search += np.multiply(self.k_diag, alpha_search)

        obj_search = self.log_joint(f_search, alpha_search)

        if min_obj > obj_search:
            opt_step = step_size
            min_obj = obj_search
        step_size = self.tau * step_size

        t = t + 1

        return obj_prev, min_obj, delta_alpha,\
            step_size, max_it, t, opt_step

    def log_joint(self, f=None, alpha=None):

        """
        Evaluates objective function (negative log likelihood plus GP penalty)
        Args:
            f (): function values (if not same as class variable)
            alpha (): alpha (if not same as class variable)

        Returns:
        """

        if self.obs_idx is not None:
            f_lim = f[self.obs_idx]
            alpha_lim = alpha[self.obs_idx]
            mu_lim = self.mu[self.obs_idx]
            like = -np.sum(self.likelihood.log_like(f_lim, self.y))
            like = np.where(np.isnan(like), np.ones_like(like)*1e12, like)
            return like +\
                0.5 * np.sum(np.multiply(alpha_lim, f_lim - mu_lim))

        return -np.sum(self.likelihood.log_like(f, self.y)) +\
            0.5 * np.sum(np.multiply(alpha, f - self.mu))

    def marginal(self, kernel):
        """
        calculates marginal likelihood
        Args:
            Ks_new: new covariance if needed
        Returns: np.array for marginal likelihood

        """

        if kernel.params is not None:
            self.Ks = self.construct_Ks(self.kernel)
            self.alpha = np.zeros([self.X.shape[0]])
            self.W = np.zeros([self.X.shape[0]])
            self.grads = np.zeros([self.X.shape[0]])
            self.f = self.mu
            self.f_pred = self.f
            self.run(10)

        Ks = self.Ks
        eigs = [np.expand_dims(np.linalg.eig(K)[0], 1) for K in Ks]
        eig_K = np.squeeze(kron_list(eigs))
        self.eig_K = eig_K

        if self.obs_idx is not None:
            f_lim = self.f[self.obs_idx]
            alpha_lim = self.alpha[self.obs_idx]
            mu_lim = self.mu[self.obs_idx]
            W_lim = self.W[self.obs_idx]
            eig_k_lim = eig_K[self.obs_idx]

            pen = -0.5 * np.sum(np.multiply(alpha_lim,
                                       f_lim - mu_lim))
            pen = np.where(np.isnan(pen), np.zeros_like(pen), pen)
            eigs = 0.5 * np.sum(np.log(1 + np.multiply(eig_k_lim,
                                       W_lim)))
            eigs = np.where(np.isnan(eigs), np.zeros_like(eigs), eigs)
            like = np.sum(self.likelihood.log_like(f_lim, self.y))
            like = np.where(np.isnan(like), np.zeros_like(like), like)

            return -(pen+eigs+like)

        pen = -0.5 * np.sum(np.multiply(self.alpha,
                                   self.f - self.mu))
        eigs = - 0.5*np.sum(np.log(1 +
                                   np.multiply(eig_K, self.W)))
        like = np.sum(self.likelihood.log_like(self.f, self.y))

        return -(pen+eigs+like)

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
        if self.obs_idx is not None:
            root_K = self.root_eigdecomp[self.obs_idx, :]
        else:
            root_K = self.root_eigdecomp

        var = np.zeros([self.m])
        diag = kron_list_diag(self.Ks)

        for i in range(n_s):
            g_m = np.random.normal(size = self.m)
            g_n = np.random.normal(size = self.n)
            right_side = np.dot(root_K, g_m) +\
                         np.sqrt(self.noise)*g_n

            r = self.opt.cg(self.Ks, right_side)
            if self.obs_idx is not None:
                Wr = np.zeros(self.m)
                Wr[self.obs_idx] = r
            else:
                Wr = r
            var += np.square(kron_mvp(self.Ks, Wr))

        return np.clip(diag - var/n_s, 0, 1e12).flatten()

    def predict_mean(self, x_new):

        k_dims = [self.kernel.eval(self.kernel.params,
                                   np.expand_dims(np.unique(self.X[:, d]), 1),
                                   np.expand_dims(x_new[:, d], 1))
                  for d in self.X.shape[1]]
        kx = np.squeeze(kron_list(k_dims))
        mean = np.sum(np.multiply(kx, self.alpha)) + self.mu[0]

        return mean

    def cg_prod(self, Ks, p):
        """

        Args:
            p (): potential solution to linear system

        Returns: product Ap (left side of linear system)

        """

        if self.precondition is None:
            return p + np.multiply(np.sqrt(self.W),
                                   kron_mvp(Ks,
                                            np.multiply(np.sqrt(self.W), p)))

        Cp = np.multiply(self.precondition, p)
        noise = np.multiply(np.multiply(self.precondition,
                                        np.multiply(self.W, self.k_diag)), Cp)
        wkw = np.multiply(np.multiply(self.precondition, np.sqrt(self.W)),
                          kron_mvp(Ks, np.multiply(np.sqrt(self.W), Cp)))

        return noise + wkw + np.multiply(self.precondition, Cp)

    def gather_derivs(self):
        """

        Returns: sum of gradients, hessians

        """

        obs_f = self.f[self.obs_idx]
        obs_grad = self.grad_func(obs_f, self.y)
        obs_hess = self.hess_func(obs_f, self.y)

        agg_grad = np.zeros(self.m, np.float64)
        agg_hess = np.zeros(self.m, np.float64)

        for i, j in enumerate(self.obs_idx):
            agg_grad[j] += obs_grad[i]
            agg_hess[j] += obs_hess[i]

        return agg_grad, agg_hess

    def update_derivs(self):

        if self.obs_idx is None:
            self.grads = self.grad_func(self.f, self.y)
            hess = self.hess_func(self.f, self.y)
            self.hess = hess
            self.W = np.clip(-hess, 1e-9, 1e16)
        else:
            self.grads, hess = self.gather_derivs()
            self.hess = hess
            self.W = np.clip(-hess, 1e-9, 1e16)
        self.W = np.where(np.isnan(self.W), np.ones_like(self.W) * 1e-9,
                          self.W)

        return self.grads, self.hess

    def opt_kern(self):

        init_params = self.kernel.params

        return minimize(self.marginal, init_params, jac = False, method='CG')

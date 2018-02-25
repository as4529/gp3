import numpy as np
from gp3.utils.optimizers import CG, SGD
from gp3.utils.structure import kron_list, kron_mvp, kron_list_diag
from gp3.utils.transforms import softplus, inv_softplus
from .base import InfBase
from scipy.linalg import toeplitz
from scipy.special import expit
from tqdm import trange, tqdm_notebook
from copy import copy

"""
Class for Kronecker inference of GPs with Gaussian likelihood. Inspiration from GPML.

For references, see:

Wilson et al (2014),
Thoughts on Massively Scalable Gaussian Processes

Most of the notation follows R and W chapter 2

"""


class Vanilla(InfBase):

    def __init__(self,
                 X,
                 y,
                 kernels,
                 mu=None,
                 obs_idx=None,
                 opt_kernel=False,
                 noise=1e-6):
        """

        Args:
            X (np.array): data
            y (np.array): output
            kernel (): kernel function to use for inference
            obs_idx (np.array): Indices of observed points (partial grid)
            noise (float): observation noise
        """

        super(Vanilla, self).__init__(X, y, kernels, mu=mu, obs_idx=obs_idx,
                                      opt_kernel=opt_kernel, noise=noise)
        self.cg_opt = CG(self.cg_prod)
        self.eigvals = None
        self.eigvecs = None
        self.alpha = None
        self.optimizer = None

    def solve(self):
       """
       Uses linear conjugate gradients to solve for (K + noise)^{-1}y
       Returns:

       """
       self.alpha = self.cg_opt.cg(self.Ks, self.y - self.mu)
       return self.alpha

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
        return self.noise * p + kprod

    def predict_mean(self):
        """
        Predicts mean at X points

        Returns: f_pred(X)

        """
        if self.alpha is None:
            self.solve()
        if self.obs_idx is not None:
            Wt_alpha = np.zeros(self.m)
            Wt_alpha[self.obs_idx] = self.alpha
        else:
            Wt_alpha = self.alpha
        return kron_mvp(self.Ks, Wt_alpha)

    def marginal(self):
        if self.alpha is None:
            self.solve()
        if self.eigvals is None:
            self.eig_decomp()
        det = 0.5 * np.sum(np.log(kron_list_diag(self.eigvals) + self.noise))
        fit = 0.5 * np.dot(self.y - self.mu, self.alpha)
        return - det - fit

    def eig_decomp(self):
        """
        Calculates eigendecomposition of Kernel matrix.

        Returns: Eigendecomposition t of kernel matrix

        """
        eigvecs = []
        eigvals = []
        for e, v in self.K_eigs:
            eigvecs.append(np.real(v))
            eigvals.append(np.diag(np.real((e))))
        self.eigvecs = eigvecs
        self.eigvals = eigvals
        return self.eigvecs, self.eigvals

    def optimize_step(self, k_params, n_params):
        if self.optimizer is None:
            self.optimizer = SGD()

        # Optimizing kernel hyperparameters
        for d in range(self.d):
            k_d_params = k_params[d] if k_params[d] is not None else None
            grad_kern = np.clip(self.grad_marginal_k(d),
                                - self.max_grad, self.max_grad)
            self.kernels[d].params, k_d_params = \
                self.optimizer.step((self.kernels[d].params, grad_kern),
                                    k_d_params)
            k_params[d] = k_d_params
        # Optimizing observation noise
        noise_trans = inv_softplus(self.noise)
        grad_noise = np.clip(self.grad_marginal_noise(),
                             - self.max_grad, self.max_grad)
        grad_noise_trans = expit(noise_trans) * grad_noise
        noise_trans, n_params = \
            self.optimizer.step((noise_trans, grad_noise_trans),
                                n_params)
        self.noise = softplus(noise_trans)
        n_params = n_params

        # updating kernel and calculating loss
        self.construct_Ks()
        self.solve()
        loss = -self.marginal()
        return k_params, n_params, loss

    def optimize(self, its=100, notebook_mode=True):
        k_params = [None for _ in range(self.d)]
        n_params = None
        losses =  [-self.marginal()]
        if notebook_mode == True:
            t = tqdm_notebook(range(its), leave=True)
        else:
            t = trange(its, leave=True)

        for i in t:
            k_adam, n_adam, loss = \
                     self.optimize_step(k_params, n_params)
            losses.append(loss)
            t.set_description("Loss: " + '{0:.2f}'.format(loss))
        return losses

    def grad_marginal_k(self, d):
        n_params = len(self.kernels[d].params)
        grads = np.zeros(n_params)
        grad_K = np.squeeze(self.kernel_opts[d](self.kernels[d].params,
                                                self.X_dims[d][0],
                                                self.X_dims[d]))
        for i in range(n_params):
            K_grad_params = toeplitz(grad_K[:, i])
            Ks_grads = copy(self.Ks)
            Ks_grads[d] = K_grad_params
            grad_i = - 0.5 * self.stochastic_trace(Ks_grads) + \
                       0.5 * np.dot(self.alpha, kron_mvp(Ks_grads, self.alpha))
            grads[i] = grad_i
        return grads

    def stochastic_trace(self, Kgrad, n_s=1):
        rs = np.random.choice([-1, 1], (n_s, self.n))
        if self.obs_idx is not None:
            r_full = np.zeros((n_s, self.m))
            r_full[:, self.obs_idx] = rs
            rs = r_full
        trace = 0.
        for i in range(n_s):
            r_i = rs[i, :]
            if self.obs_idx is not None:
                Kgr = kron_mvp(Kgrad, r_i)[self.obs_idx]
                trace += np.dot(r_i[self.obs_idx],
                                self.cg_opt.cg(self.Ks, Kgr))
            else:
                Kgr = kron_mvp(Kgrad, r_i)
                trace += np.dot(r_i, self.cg_opt.cg(self.Ks, Kgr))
        return trace / n_s

    def grad_marginal_noise(self, n_s=1):

        rs = np.random.choice([-1, 1], (n_s, self.n))
        trace = 0.
        for i in range(n_s):
            r_i = rs[i, :]
            trace += np.dot(r_i, self.cg_opt.cg(self.Ks, r_i))
        trace = trace / n_s
        return np.array([- 0.5 * trace + 0.5 * np.dot(self.alpha, self.alpha)])

    def variance(self, n_s=30):
        """
        Stochastic approximator of predictive variance.
         Follows "Massively Scalable GPs"
        Args:
            n_s (int): Number of iterations to run stochastic approximation

        Returns: Approximate predictive variance at grid points

        """

        if self.eigvals is None:
            self.eig_decomp()

        Q = self.eigvecs
        Q_t = [v.T for v in self.eigvecs]
        Vr = [np.sqrt(e) for e in self.eigvals]

        var = np.zeros([self.m])
        diag = kron_list_diag(self.Ks)
        g_m = np.random.normal(size=(n_s, self.m))
        g_n = np.random.normal(size=(n_s, self.n))

        for i in range(n_s):
            Kroot_g = kron_mvp(Q, kron_mvp(Vr, kron_mvp(Q_t, g_m[i,:])))
            if self.obs_idx is not None:
                Kroot_g = Kroot_g[self.obs_idx]
            right_side = Kroot_g + np.sqrt(self.noise) * g_n[i,:]
            r = self.cg_opt.cg(self.Ks, right_side)
            if self.obs_idx is not None:
                Wr = np.zeros(self.m)
                Wr[self.obs_idx] = r
            else:
                Wr = r
            var += np.square(kron_mvp(self.Ks, Wr))
        return np.clip(diag - var/n_s, 0, a_max = None).flatten()

    def variance_slow(self, n_s=30):

        K_uu = kron_list(self.Ks)
        K_xx = K_uu
        K_ux = K_uu

        if self.obs_idx is not None:
            K_ux = K_uu[:, self.obs_idx]
            K_xx = K_uu[self.obs_idx, :][:, self.obs_idx]
            self.K_ux = K_ux
        A = K_xx + np.diag(np.ones(self.n) * self.noise)
        A_inv = np.linalg.inv(A)
        A_inv_chol = np.linalg.cholesky(A_inv)
        var = np.zeros([self.m])

        for i in range(n_s):
            eps = np.random.normal(0, 1, size = self.n)
            r = np.dot(A_inv_chol, eps)
            var += np.square(np.dot(K_ux, r))
        return np.clip(np.diag(K_uu) - var / n_s, 0, a_max = None).flatten()

    def variance_exact(self):

        K_uu = kron_list(self.Ks)
        K_xx = K_uu
        K_ux = K_uu
        if self.obs_idx is not None:
            K_ux = K_uu[:, self.obs_idx]
            K_xx = K_uu[self.obs_idx, :][:, self.obs_idx]

        A = K_xx + np.diag(np.ones(self.n) * self.noise)
        A_inv = np.linalg.inv(A)
        return np.diag(K_uu - np.dot(K_ux, A_inv).dot(K_ux.T))
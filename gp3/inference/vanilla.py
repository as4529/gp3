import numpy as np
from gp3.utils.optimizers import CG, SGD
from gp3.utils.structure import kron_list, kron_mvp, kron_list_diag
from gp3.utils.transforms import softplus, inv_softplus
from .base import InfBase
from scipy.linalg import toeplitz
from scipy.special import expit
from tqdm import trange, tqdm_notebook
from copy import copy
from autograd import jacobian

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
                 noise=1e-6,
                 opt_idx=None):
        """

        Args:
            X (np.array): data
            y (np.array): output
            kernel (): kernel function to use for inference
            obs_idx (np.array): Indices of observed points (partial grid)
            noise (float): observation noise
        """

        super(Vanilla, self).__init__(X, y, kernels, mu=mu, obs_idx=obs_idx,
                                      noise=noise)
        self.cg_opt = CG(self.A_prod)
        self.mvp = kron_mvp
        self.eigvals = None
        self.eigvecs = None
        self.alpha = None
        self.optimizer = None
        self.kernel_grads = None
        self.opt_idx = range(self.d) if opt_idx is None else opt_idx

    def solve(self):
       """
       Uses linear conjugate gradients to solve for (K + noise)^{-1}y
       Returns:

       """
       mu = self.mu
       if self.obs_idx is not None:
           mu = mu[self.obs_idx]
       self.alpha = self.cg_opt.cg(self.Ks, self.y - mu)
       return

    def A_prod(self, Ks, p):
        """

        Args:
            p (np.array): potential solution to linear system

        Returns: product Ap (left side of linear system)

        """
        kprod = self.K_prod(Ks, p)
        return self.noise * p + kprod

    def K_prod(self, Ks, p):
        """
        Product with covariance matrix K
        Args:
            Ks (): kronecker decomposition of K
            p (): vector for product

        Returns:

        """
        if self.obs_idx is not None:
            Wp = np.zeros(self.m)
            Wp[self.obs_idx] = p
            kprod = self.mvp(Ks, Wp)[self.obs_idx]
        else:
            kprod = self.mvp(Ks, p)
        return kprod

    def predict_mean(self):
        """
        Predicts mean at X points

        Returns: f_pred(X)

        """
        if self.alpha is None:
            self.solve()
        if self.obs_idx is not None:
            Wp = np.zeros(self.m)
            Wp[self.obs_idx] = self.alpha
            kprod = self.mvp(self.Ks, Wp)
        else:
            kprod = self.mvp(self.Ks, self.alpha)
        return self.mu + kprod

    def marginal(self):
        """
        Calculates marginal likelihood
        Returns: marginal likelihood

        """
        if self.alpha is None:
            self.solve()
        if self.eigvals is None:
            self.eig_decomp()
        mu = self.mu
        if self.obs_idx is not None:
            mu = self.mu[self.obs_idx]
        det = 0.5 * np.sum(np.log(kron_list_diag(self.eigvals) + self.noise))
        fit = 0.5 * np.dot(self.y - mu, self.alpha)
        prior = 0
        for kern in self.kernels:
            prior += kern.log_prior(kern.params)
        return - det - fit - prior

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
        return

    def optimize_step(self, k_params, n_params, update=True):
        """
        Runs one step of optimization
        Args:
            k_params (): optimizer params for kernel optimization
            n_params (): optimizer params for noise uptimization
            update (): whether or not to update parameters

        Returns: updated k_params, n_params

        """

        if self.opt_idx is None:
            self.opt_idx = list(range(self.d))
        if self.optimizer is None:
            self.optimizer = SGD()
        if self.kernel_grads is None:
            self.kernel_grads = []
            for i in self.opt_idx:
                self.kernel_grads.append((jacobian(self.kernels[i].eval),
                                          jacobian(self.kernels[i].log_prior)))

        # Optimizing kernel hyperparameters
        for i, d in enumerate(self.opt_idx):
            k_d_params = k_params[i] if k_params[i] is not None else None
            grad_kern_marginal = np.clip(self.grad_marginal_k(i, d),
                                - self.max_grad, self.max_grad)
            grad_kern_penalty = np.clip(self.grad_penalty_k(i, d),
                                - self.max_grad, self.max_grad)
            grad_kern = grad_kern_marginal - grad_kern_penalty
            self.kernels[d].params, k_d_params = \
                self.optimizer.step((self.kernels[d].params, grad_kern),
                                    k_d_params)
            k_params[i] = k_d_params

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
        loss = -self.marginal()
        if update:
            self.solve()
        return k_params, n_params, loss

    def optimize(self, its=100, notebook_mode=True):
        """
        Kernel optimization
        Args:
            its (): maximum number of iterations
            notebook_mode (): for tqdm notebook

        Returns: sequence of loss values

        """
        if self.opt_idx is None:
            self.opt_idx = list(range(self.d))
        self.kernel_grads = []
        for i in self.opt_idx:
            self.kernel_grads.append((jacobian(self.kernels[i].eval),
                                      jacobian(self.kernels[i].log_prior)))

        k_params = [None for _ in range(len(self.opt_idx))]
        n_params = None
        losses =  [-self.marginal()]
        if notebook_mode == True:
            t = tqdm_notebook(range(its), leave=True)
        else:
            t = trange(its, leave=True)

        for i in t:
            k_params, n_params, loss = \
                     self.optimize_step(k_params, n_params)
            losses.append(loss)
            t.set_description("Loss: " + '{0:.2f}'.format(loss))
            if self.loss_check(losses):
                break
        return losses

    def grad_marginal_k(self, i, d):
        """
        Gradient of marginal likelihood w.r.t. kernel hyperparameters
        Args:
            i (): index of kernel in self.kernels
            d (): dimension of kernel

        Returns:
            gradient of marginal w.r.t. ith kernel at dth dimension

        """
        n_params = len(self.kernels[d].params)
        grads = np.zeros(n_params)
        grad_K = np.squeeze(self.kernel_grads[i][0](self.kernels[d].params,
                                                 self.X_dims[d][0],
                                                 self.X_dims[d]))
        for j in range(n_params):
            K_grad_params = toeplitz(grad_K[:, j])
            Ks_grads = copy(self.Ks)
            Ks_grads[d] = K_grad_params
            grad_j = - 0.5 * self.stochastic_trace(Ks_grads) + \
                       0.5 * np.dot(self.alpha, self.K_prod(Ks_grads, self.alpha))
            grads[j] = grad_j
        return grads

    def grad_penalty_k(self, i, d):
        """
        Gradient of kernel prior w.r.t kernel parameters
        Args:
            i (): index of kernel in self.kernels
            d (): dimension of kernel

        Returns: gradient

        """
        return self.kernel_grads[i][1](self.kernels[d].params)

    def stochastic_trace(self, Kgrad, n_s=1):
        """
        Stochastic estimator of trace term for gradient
        Args:
            Kgrad (): gradient w.r.t kernel matrix
            n_s (): number of samples for estimator

        Returns: estimate of trace term

        """
        rs = np.random.choice([-1, 1], (n_s, self.n))
        if self.obs_idx is not None:
            r_full = np.zeros((n_s, self.m))
            r_full[:, self.obs_idx] = rs
            rs = r_full
        trace = 0.
        for i in range(n_s):
            r_i = rs[i, :]
            if self.obs_idx is not None:
                Kgr = self.mvp(Kgrad, r_i)[self.obs_idx]
                trace += np.dot(r_i[self.obs_idx],
                                self.cg_opt.cg(self.Ks, Kgr))
            else:
                Kgr = self.mvp(Kgrad, r_i)
                trace += np.dot(r_i, self.cg_opt.cg(self.Ks, Kgr))
        return trace / n_s

    def grad_marginal_noise(self, n_s=1):
        """
        Gradient of marginal likelihood w.r.t observation noise
        Args:
            n_s (): Number of samples

        Returns:

        """
        rs = np.random.choice([-1, 1], (n_s, self.n))
        trace = 0.
        for i in range(n_s):
            r_i = rs[i, :]
            trace += np.dot(r_i, self.cg_opt.cg(self.Ks, r_i))
        trace = trace / n_s
        return np.array([-0.5 * trace + 0.5 * np.dot(self.alpha, self.alpha)])

    def variance_pmap(self, n_s=30):
        """
        Stochastic approximator of predictive variance.
         Follows "Massively Scalable GPs"
        Args:
            n_s (int): Number of iterations to run stochastic approximation

        Returns: Approximate predictive variance at grid points

        """
        if self.eigvals or self.eigvecs is None:
            self.eig_decomp()

        Q = self.eigvecs
        Q_t = [v.T for v in self.eigvecs]
        Vr = [np.nan_to_num(np.sqrt(e)) for e in self.eigvals]

        diag = kron_list_diag(self.Ks) + self.noise
        samples = []

        for i in range(n_s):
            g_m = np.random.normal(size=self.m)
            g_n = np.random.normal(size=self.n)

            Kroot_g = kron_mvp(Q, kron_mvp(Vr, kron_mvp(Q_t, g_m)))
            if self.obs_idx is not None:
                Kroot_g = Kroot_g[self.obs_idx]
            right_side = Kroot_g + np.sqrt(self.noise) * g_n

            r = self.cg_opt.cg(self.Ks, right_side)
            if self.obs_idx is not None:
                Wr = np.zeros(self.m)
                Wr[self.obs_idx] = r
            else:
                Wr = r
            samples.append(kron_mvp(self.Ks, Wr))

        est = np.var(samples, axis=0)
        return np.clip(diag - est, 0, a_max = None).flatten()

    def variance_exact(self):
        """
        Exact computation of variance
        Returns: exact variance

        """
        K_uu = kron_list(self.Ks)
        K_xx = K_uu
        K_ux = K_uu
        if self.obs_idx is not None:
            K_ux = K_uu[:, self.obs_idx]
            K_xx = K_uu[self.obs_idx, :][:, self.obs_idx]

        A = K_xx + np.diag(np.ones(self.n) * self.noise)
        A_inv = np.linalg.inv(A)
        return np.diag(K_uu - np.dot(K_ux, A_inv).dot(K_ux.T)) + self.noise

    def loss_check(self, losses):
        """
        Checks conditions for loss decreasing

        Returns: True if condition satisfied

        """
        if len(losses) < 10:
            return False
        if sum(x <= y for x, y in zip(losses[-10:], losses[-9:])) > 5 and\
            losses[-1] - losses[-10] < 1e-3*abs(losses[-10]):
            return True
import autograd.numpy as np
from autograd import elementwise_grad as egrad, jacobian
from gp3.utils.structure import kron_mvp, kron_list_diag
from gp3.utils.optimizers import Adam
from tqdm import trange, tqdm_notebook
from copy import deepcopy
from .base import InfBase

"""
Stochastic Variational Inference for Gaussian Processes with
 Non-Gaussian Likelihoods
"""



class SVIBase(InfBase):
    """
    Base class for stochastic variational inference.
    """
    def __init__(self,
                 X,
                 y,
                 kernel,
                 likelihood,
                 mu = None,
                 obs_idx = None,
                 opt_kernel = False,
                 noise = 1e-2,
                 optimizer = Adam()):

        super(SVIBase, self).__init__(X, y, kernel, likelihood,
                                      mu, obs_idx, opt_kernel, noise=noise)

        self.elbos = []
        self.q_mu = self.mu
        self.likelihood = likelihood
        self.likelihood_opt = egrad(self.likelihood.log_like)
        self.optimizer = optimizer

    def predict(self):
        """
        GP predictions
        Returns: predictions
        """
        Ks = []
        for d in range(self.X.shape[1]):
            K = self.kernels[d].eval(self.kernels[d].params,
                                 np.expand_dims(np.unique(self.X[:, d]), 1))
            Ks.append(K)

        f_pred = kron_mvp(Ks, kron_mvp(self.K_invs, self.q_mu))

        return f_pred

class MFSVI(SVIBase):
    """
    Stochastic variational inference with mean-field variational approximation
    """
    def __init__(self, X, y, kernel, likelihood,
                 mu = None, obs_idx = None, opt_kernel = False):
        """
        Args:
            kernel (): kernel function
            likelihood (): likelihood function. Requires log_like() function
            X (): data
            y (): responses
            mu (): prior mean
            noise (): noise variance
            obs_idx (): if dealing with partial grid, indices of grid that are observed
            verbose (): print or not
        """

        super(MFSVI, self).__init__(X, y, kernel, likelihood,
                                    mu, obs_idx, opt_kernel)

        self.q_S = np.ones(self.m) * np.log(self.Ks[0][0, 0] ** self.d)
        self.mu_params, self.s_params, self.k_params = (None for _ in range(3))

    def run(self, its, n_samples=1, notebook_mode = True):
        """
        Runs stochastic variational inference
        Args:
            its (int): Number of iterations
            n_samples (int): Number of samples for SVI
        Returns: Nothing, but updates instance variables
        """

        if notebook_mode == True:
            t = tqdm_notebook(range(its), leave=False)
        else:
            t = trange(its, leave = False)

        for i in t:

            KL_grad_S, KL_grad_mu = self.grad_KL_S(), self.grad_KL_mu()
            grads_mu, grads_S, es, rs= ([] for i in range(4))

            for j in range(n_samples):
                eps = np.random.normal(size=self.m)
                r = self.q_mu + np.multiply(np.sqrt(np.exp(self.q_S)), eps)
                like_grad_S, like_grad_mu = self.grad_like(r, eps)

                grad_mu = np.clip(-KL_grad_mu + like_grad_mu,
                                  -self.max_grad, self.max_grad)
                grad_S = np.clip(-KL_grad_S + like_grad_S,
                                 -self.max_grad, self.max_grad)

                grads_mu.append(grad_mu)
                grads_S.append(grad_S)
                es.append(eps)
                rs.append(r)

            obj, kl, like = self.eval_obj(self.q_S, self.q_mu, rs)
            self.elbos.append(-obj)

            t.set_description("ELBO: " + '{0:.2f}'.format(-obj) +
                              " | KL: " + '{0:.2f}'.format(kl) +
                              " | logL: " + '{0:.2f}'.format(like))

            S_vars= (self.q_S, np.mean(grads_S, 0))
            mu_vars = (self.q_mu, np.mean(grads_mu, 0))

            self.q_mu, m_mu, v_mu = self.optimizer.step(mu_vars, self.mu_params)
            self.q_S, m_s, v_s = self.optimizer.step(S_vars, self.s_params)
            if i > 100 and self.loss_check() == True:
                print("converged at", i, "iterations")
                return

        print("warning: inference may not have converged")

        return

    def eval_obj(self, S, q_mu, rs, kern_params = None):
        """
        Evaluates variational objective
        Args:
            S (): Variational variances
            q_mu (): Variational mean
            r (): Transformed random sample
        Returns: ELBO evaluation
        """
        objs, kls, likes = ([] for i in range(3))
        kl = self.KLqp(S, q_mu, kern_params)

        for r in rs:

            if self.obs_idx is not None:
                r_obs = r[self.obs_idx]
            else:
                r_obs = r

            like = np.sum(self.likelihood.log_like(r_obs, self.y))
            obj = kl - like
            objs.append(obj)
            likes.append(like)

        return np.mean(objs), kl, np.mean(likes)

    def KLqp(self, S, q_mu, kern_params):
        """
        Calculates KL divergence between q and p
        Args:
            S (): Variational variances
            q_mu (): Variational mean
        Returns: KL divergence between q and p
        """

        if kern_params is None:
            K_invs = self.K_invs
            k_inv_diag = self.k_inv_diag
            det_K = self.det_K
        else:
            Ks, K_invs = self.construct_Ks()
            k_inv_diag = kron_list_diag(K_invs)
            det_K = self.log_det_K(Ks)

        k_inv_mu = kron_mvp(K_invs, self.mu - q_mu)
        mu_penalty = np.sum(np.multiply(self.mu -q_mu, k_inv_mu))
        det_S = np.sum(S)
        trace_term = np.sum(np.multiply(k_inv_diag, np.exp(S)))

        kl = 0.5 * (det_K - self.m - det_S +
                    trace_term + mu_penalty)

        return max(0, kl)

    def grad_KL_S(self):
        """
        Natural gradient of KL divergence w.r.t variational variances
        Returns: returns gradient
        """
        euc_grad = 0.5 * (-1. + np.multiply(self.k_inv_diag, np.exp(self.q_S)))

        return 2 * euc_grad / self.m

    def grad_KL_mu(self):
        """
        Natural gradient of KL divergence w.r.t variational mean
        Returns: returns gradient
        """
        return np.multiply(np.exp(self.q_S), -kron_mvp(self.K_invs, self.mu - self.q_mu))

    def grad_like(self, r, eps):
        """
        Gradient of likelihood w.r.t variational parameters
        Args:
            r (): Transformed random sample
            eps (): Random sample
        Returns: gradient w.r.t variances, gradient w.r.t mean
        """
        if self.obs_idx is not None:
            r_obs = r[self.obs_idx]
        else:
            r_obs = r

        dr = self.likelihood_opt(r_obs, self.y)
        dr[np.isnan(dr)] = 0.

        if self.obs_idx is not None:
            grad_mu = np.zeros(self.m)
            grad_mu[self.obs_idx] = dr
        else:
            grad_mu = dr
        grad_S = np.multiply(grad_mu, np.multiply(eps,
                                      np.multiply(0.5/np.sqrt(np.exp(self.q_S)),
                                                  np.exp(self.q_S))))

        return grad_S, grad_mu

    def predict(self):
        """
        GP predictions
        Returns: predictions
        """
        Ks = [self.kernel.eval(self.kernel.params, X_dim) for X_dim in self.X_dims]

        f_pred = kron_mvp(Ks, kron_mvp(self.K_invs, self.q_mu))

        return f_pred

    def sample_post(self, n_samples = 1):
        """
        Sampels from the variational posterior
        Args:
            n_samples (int): Number of desired samples

        Returns: Sample(s) from variational posterior

        """

        return self.q_mu + \
               np.multiply(np.expand_dims(np.sqrt(np.exp(self.q_S)), 1),
                           np.random.normal(size = (self.m, n_samples))).flatten()

    def loss_check(self):
        """
        Checks conditions for loss decreasing

        Returns: True if condition satisfied

        """
        if sum(x >= y for x, y in zip(self.elbos[-100:], self.elbos[-99:])) > 50 and\
            self.elbos[-1] - self.elbos[-100] < 1e-3*abs(self.elbos[-100]):
            return True

class FullSVI(SVIBase):
    """
    Variational inference with full variational covariance matrix
    """

    def __init__(self, X, y, kernel, likelihood,
                 mu = None, obs_idx=None):
        """
        Args:
            function
            X (): data
            y (): responses
            kernel (): kernel function
            likelihood (): likelihood function. Requires log_like()
            mu (): prior mean
            obs_idx (): if dealing with partial grid, indices of grid that are observed
        """

        super(FullSVI, self).__init__(X, y, kernel, likelihood, mu, obs_idx)

        self.Rs = self.initialize_Rs_prior()
        self.trace_term, self.traces = self.calc_trace_term()
        self.v_mu, self.v_k, self.m_mu, self.m_k = (None for _ in range(4))
        self.v_Rs = [None for _ in range(self.d)]
        self.m_Rs = [None for _ in range(self.d)]

    def run(self, its, notebook_mode = True):
        """
        Runs stochastic variational inference
        Args:
            its (): Number of iterations
        Returns: Nothing, but updates instance variables
        """

        if notebook_mode == True:
            t = tqdm_notebook(range(its), leave=False)
        else:
            t = trange(its, leave = False)

        for i in t:
            self.trace_term, self.traces = self.calc_trace_term()
            KL_grad_R = self.grad_KL_R()
            KL_grad_mu = self.grad_KL_mu()

            eps = np.random.normal(size = self.m)
            r = self.q_mu + kron_mvp(self.Rs, eps)
            like_grad_R, like_grad_mu = self.grad_like(r, eps)

            grad_R = [np.clip(-KL_grad_R[i] + like_grad_R[i], -self.max_grad,
                              self.max_grad)
                      for i in range(len(KL_grad_R))]
            grad_mu= np.clip(-KL_grad_mu + like_grad_mu, -self.max_grad,
                             self.max_grad)
            R_and_grads = list(zip(self.Rs, grad_R))
            mu_and_grad = (self.q_mu, grad_mu)

            obj, kl, like = self.eval_obj(self.Rs, self.q_mu, r)
            self.elbos.append(-obj)

            self.q_mu, self.m_mu, self.v_mu =\
                self.optimizer.step(mu_and_grad, self.m_mu,
                                    self.v_mu, i + 1)

            for d in range(self.d):
                self.Rs[d], self.m_Rs[d], self.v_Rs[d] =\
                    self.optimizer.step(R_and_grads[d], self.m_Rs[d],
                                        self.v_Rs[d], i + 1)

            t.set_description("ELBO: " + '{0:.2f}'.format(-obj) +
                              " | KL: " + '{0:.2f}'.format(kl) +
                              " | logL: " + '{0:.2f}'.format(like))

        return

    def eval_obj(self, Rs, q_mu, r):
        """
        Evaluates variational objective
        Args:
            Rs (): Variational covariances
             (Cholesky decomposition of Kronecker decomp)
            q_mu (): Variational mean
            r (): Transformed random sample
        Returns: ELBO evaluation
        """
        kl = self.KL_calc(Rs, q_mu)
        if self.obs_idx is not None:
            r_obs = r[self.obs_idx]
        else:
            r_obs = r
        like = np.sum(self.likelihood.log_like(r_obs, self.y))
        obj = kl - like

        return obj, kl, like

    def KL_calc(self, Rs, q_mu):
        """
        Calculates KL divergence between q and p
        Args:
            Rs (): Variational covariance
            q_mu (): Variational mean
        Returns: KL divergence between q and p
        """
        k_inv_mu = kron_mvp(self.K_invs, self.mu - q_mu)
        mu_penalty = np.sum(np.multiply(self.mu -q_mu, k_inv_mu))
        det_S = self.log_det_S(Rs)
        trace_term = self.calc_trace_term(Rs)[0]
        kl = 0.5 * (self.det_K - self.m - det_S +
                      trace_term + mu_penalty)

        if kl < 0:
            return 0.

        return max(0, kl)

    def grad_KL_R(self):
        """
        Gradient of KL divergence w.r.t variational covariance
        Returns: returns gradient
        """
        return [np.diag(-2*self.m/self.Rs[d].shape[0]/
                         np.diag(self.Rs[d])) +\
                         np.prod(self.traces)/self.traces[d] *
                         np.dot(self.K_invs[d], self.Rs[d])
                         for d in range(len(self.Rs))]

    def grad_KL_mu(self):
        """
        Gradient of KL divergence w.r.t variational mean
        Returns: returns gradient
        """
        return -kron_mvp(self.K_invs, self.mu - self.q_mu)

    def grad_like(self, r, eps):
        """
        Gradient of likelihood w.r.t variational parameters
        Args:
            r (): Transformed random sample
            eps (): Random sample
        Returns: gradient w.r.t covariance, gradient w.r.t mean
        """
        if self.obs_idx is not None:
            r_obs = r[self.obs_idx]
        else:
            r_obs = r


        dr = self.likelihood_opt(r_obs, self.y)
        dr[np.isnan(dr)] = 0.
        grads_R = []

        for d in range(len(self.Rs)):

            Rs_copy = deepcopy(self.Rs)
            n = Rs_copy[d].shape[0]
            grad_R = np.zeros((n, n))

            for i, j in zip(*np.triu_indices(n)):
                R_d = np.zeros((n, n))
                R_d[i, j] = 1.
                Rs_copy[d] = R_d
                dR_eps = kron_mvp(Rs_copy, eps)
                if self.obs_idx is not None:
                    dR_eps = dR_eps[self.obs_idx]
                grad_R[i, j] = np.sum(np.multiply(dr, dR_eps))
            grads_R.append(grad_R)

        if self.obs_idx is not None:
            grad_mu = np.zeros(self.m)
            grad_mu[self.obs_idx] = dr
        else:
            grad_mu = dr

        return grads_R, grad_mu

    def line_search(self, Rs_grads, mu_grads, obj_init, r, eps):
        """
        Performs line search to find optimal step size
        Args:
            Rs_grads (): Gradients of R (variational covariances)
            mu_grads (): Gradients of mu (variational mean)
            obj_init (): Initial objective value
            r (): transformed random Gaussian sample
            eps (): random Gaussian sample
        Returns: Optimal step size
        """
        step = 1.

        while step > 1e-9:

            R_search = [R + step * R_grad
                        for R_grad, R in Rs_grads]
            mu_search = mu_grads[1] + step * mu_grads[0]
            r_search = mu_search + kron_mvp(R_search, eps)
            obj_search, kl_search, like_search = self.eval_obj(R_search, mu_search,
                                                               r_search)
            if obj_init - obj_search > step:
                return R_search, mu_search, obj_search, step

            step = step * 0.5

        return None

    def calc_trace_term(self, Rs = None):
        """
        Calculates trace term for objective function
        Args:
            Rs (): trace of variational covariance, and individual trace over dimensions
        Returns:
        """
        if Rs is None:
            Rs = self.Rs

        traces = [np.trace(np.dot(self.K_invs[d],
                                     np.dot(np.transpose(Rs[d]), Rs[d])))
                                     for d in range(len(self.K_invs))]

        return np.prod(traces), traces

    def log_det_S(self, Rs = None):
        """
        Log determinant of variational covariance
        Args:
            Rs (): Kronecker decomposed variational covariance
        Returns: determinant
        """
        if Rs is None:
            Rs = self.Rs

        return 2*np.sum([self.m/R.shape[0]*
                                 np.sum(np.log(np.diag(R)))
                                 for R in Rs])

    def initialize_Rs(self):
        """
        Initializes upper triangular decomp
         of kronecker decomp of vairational covariance
        using identity matrix
        Returns: Rs (identity matrices)
        """
        return[np.eye(K.shape[0])
            for K in self.Ks]

    def initialize_Rs_prior(self):
        """
        Initializes Rs using cholesky decomps of prior covariance
        Returns: Rs (cholesky decomp of prior covariances)
        """
        return [np.transpose(np.linalg.cholesky(K))
                for K in self.Ks]


    def cg_prod(self, A, x):
        """
        Not currently used, but for conjugate gradient method
        Args:
            A ():
            x ():
        Returns:
        """
        return kron_mvp(A, x)

    def sample_post(self):
        """
        Draws a sample from the GPLVM posterior
        Returns: sample
        """

        eps = np.random.normal(size = self.m)
        r = self.q_mu + kron_mvp(self.Rs, eps)

        return r
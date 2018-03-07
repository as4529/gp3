import autograd.numpy as np
from autograd import elementwise_grad as egrad, jacobian
from gp3.utils.structure import kron_mvp, kron_list_diag, kron_list
from gp3.utils.optimizers import SGD, Adam
from tqdm import trange, tqdm_notebook
from copy import deepcopy
from .base import InfBase
import scipy

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
                 kernels,
                 likelihood,
                 mu = None,
                 obs_idx = None,
                 noise=1e-2,
                 optimizer=Adam()):

        super(SVIBase, self).__init__(X, y, kernels, likelihood,
                                      mu, obs_idx, noise=noise)
        self.elbos = []
        self.noise = noise
        self.q_mu = self.mu
        self.likelihood = likelihood
        self.likelihood_grad = egrad(self.likelihood.log_like)
        self.optimizer = optimizer

    def predict(self):
        """
        GP predictions
        Returns: predictions
        """
        Ks = []
        for d in range(self.X.shape[1]):
            K = self.kernels[d].eval(self.kernels[d].params,
                                 self.X_dims[d])
            Ks.append(K)
        f_pred = kron_mvp(Ks, kron_mvp(self.K_invs, self.q_mu))
        return f_pred

    def construct_Ks(self, kernels=None, noise=1e-2):
        super(SVIBase, self).construct_Ks()
        self.Ks = [K + np.diag(np.ones(K.shape[0])) * noise for K in self.Ks]
        self.K_invs = [np.linalg.inv(K)
                       for K in self.Ks]
        self.k_inv_diag = kron_list_diag(self.K_invs)
        self.det_K = self.log_det_K()

    def loss_check(self):
        """
        Checks conditions for loss decreasing

        Returns: True if condition satisfied

        """
        if sum(x >= y for x, y in zip(self.elbos[-100:], self.elbos[-99:])) > 50 and\
            self.elbos[-1] - self.elbos[-100] < 1e-3*abs(self.elbos[-100]):
            return True

class MFSVI(SVIBase):
    """
    Stochastic variational inference with mean-field variational approximation
    """
    def __init__(self, X, y, kernel, likelihood,
                 mu = None, obs_idx = None):
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
                                    mu, obs_idx)
        self.q_S = np.log(kron_list_diag(self.Ks))
        self.mu_params, self.s_params = (None, None)

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
            self.q_mu, self.mu_params = self.optimizer.step(mu_vars, self.mu_params)
            self.q_S, self.s_params = self.optimizer.step(S_vars, self.s_params)
        print("warning: inference may not have converged")
        return

    def eval_obj(self, S, q_mu, rs):
        """
        Evaluates variational objective
        Args:
            S (): Variational variances
            q_mu (): Variational mean
            r (): Transformed random sample
        Returns: ELBO evaluation
        """
        objs, kls, likes = ([] for i in range(3))
        kl = self.KLqp(S, q_mu)
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

    def KLqp(self, S, q_mu):
        """
        Calculates KL divergence between q and p
        Args:
            S (): Variational variances
            q_mu (): Variational mean
        Returns: KL divergence between q and p
        """

        k_inv_mu = kron_mvp(self.K_invs, self.mu - q_mu)
        mu_penalty = np.sum(np.multiply(self.mu - q_mu, k_inv_mu))
        det_S = np.sum(S)
        trace_term = np.sum(np.multiply(self.k_inv_diag, np.exp(S)))
        kl = 0.5 * (self.det_K - self.m - det_S +
                    trace_term + mu_penalty)
        return kl

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
        return np.multiply(np.exp(self.q_S),
                           - kron_mvp(self.K_invs, self.mu - self.q_mu))

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
        dr = self.likelihood_grad(r_obs, self.y)
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

    def sample_predictive(self):

        return

class FullSVI(SVIBase):
    """
    Variational inference with full variational covariance matrix
    """

    def __init__(self, X, y, kernel, likelihood,
                 mu = None, obs_idx=None, linesearch=True):
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
        self.mu_params = None
        self.R_params = [None for _ in range(self.d)]
        self.likes = []
        self.kls = []
        self.linesearch = linesearch

    def run(self, its):
        """
        Runs stochastic variational inference
        Args:
            its (): Number of iterations

        Returns: Nothing, but updates instance variables

        """

        t = trange(its, leave=True)

        for i in t:
            self.calc_trace_term()
            KL_grad_R = self.grad_KL_R()
            KL_grad_mu = self.grad_KL_mu()

            eps = np.random.normal(size=self.n)
            r = self.q_mu + kron_mvp(self.Rs, eps)
            like_grad_R, like_grad_mu = self.grad_like(r, eps)
            grad_R = [-KL_grad_R[i] + like_grad_R[i]
                      for i in range(len(KL_grad_R))]
            grad_mu = -KL_grad_mu + like_grad_mu
            R_and_grads = list(zip(grad_R, self.Rs))
            mu_and_grad = (grad_mu, self.q_mu)

            obj, kl, like = self.eval_obj(self.Rs, self.q_mu, r)
            self.elbos.append(-obj)

            if self.linesearch:
                ls_res = self.line_search(R_and_grads, mu_and_grad, obj, r, eps)
                step = 0.
                if ls_res is not None:
                    step = ls_res[-1]
                t.set_description("ELBO: " + '{0:.2f}'.format(-obj) +
                                  " | KL: " + '{0:.2f}'.format(kl) +
                                  " | logL: " + '{0:.2f}'.format(like) +
                                  " | step: " + str(step))
                if ls_res is not None:
                    self.Rs = ls_res[0]
                    self.q_mu = ls_res[1]
            else:
                t.set_description("ELBO: " + '{0:.2f}'.format(-obj) +
                                  " | KL: " + '{0:.2f}'.format(kl) +
                                  " | logL: " + '{0:.2f}'.format(like))
                self.q_mu, self.mu_params = \
                    self.optimizer.step(mu_and_grad, self.mu_params)
                for d in range(self.d):
                    self.Rs[d], self.R_params[d] = \
                        self.optimizer.step(R_and_grads[d], self.R_params[d])
        self.f_pred = self.predict()
        return

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

        while step > 1e-15:

            R_search = [np.clip(R + step*R_grad, 0., np.max(R))
                        for (R_grad, R) in Rs_grads]
            mu_search = mu_grads[1] + step*mu_grads[0]
            r_search = mu_search + kron_mvp(R_search, eps)
            obj_search, kl_search, like_search = self.eval_obj(R_search, mu_search,
                                                               r_search)
            if obj_init - obj_search > step:
                pos_def = True
                for R in R_search:
                    if np.all(np.linalg.eigvals(R) > 0) == False:
                        pos_def = False
                if pos_def:
                    return R_search, mu_search, obj_search, step
            step = step * 0.5
        return None

    def eval_obj(self, Rs, q_mu, r):
        """
        Evaluates variational objective
        Args:
            Rs (): Variational covariances (Cholesky decomposition of Kronecker decomp)
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
        mu_penalty = np.sum(np.multiply(self.mu - q_mu, k_inv_mu))
        det_S = self.log_det_S(Rs)
        trace_term = self.calc_trace_term(Rs)[0]
        kl = 0.5 * (self.det_K - self.n - det_S +
                    trace_term + mu_penalty)
        return max(0, kl)

    def grad_KL_R(self):
        """
        Gradient of KL divergence w.r.t variational covariance
        Returns: returns gradient
        """
        grad_Rs = []
        for d in range(len(self.Rs)):
            R_d = self.Rs[d]
            n = R_d.shape[0]
            grad_R = np.zeros((n, n))
            R_inv = np.linalg.inv(R_d)
            K_inv_R = self.K_invs[d].dot(R_d)
            for i, j in zip(*np.triu_indices(n)):
                grad_R[i, j] = - R_inv[i, j] + \
                                 np.prod(self.traces) / self.traces[d] * \
                                 K_inv_R[i, j]
            grad_Rs.append(np.nan_to_num(grad_R))
        return grad_Rs

    def grad_KL_mu(self):
        """
        Gradient of KL divergence w.r.t variational mean
        Returns: returns gradient

        """
        return kron_mvp(self.K_invs, self.q_mu - self.mu)

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
        dr = self.likelihood_grad(r_obs, self.y)
        dr[np.isnan(dr)] = 0.
        self.dr = dr
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
        grad_mu = np.zeros(self.n)
        grad_mu[self.obs_idx] = dr

        return grads_R, grad_mu

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
                                  Rs[d].dot(Rs[d])))
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
        return np.sum([self.n/R.shape[0]*
                       np.linalg.slogdet(R.T.dot(R))[1]
                       for R in Rs])

    def log_det_K(self, Ks=None):
        """
        Log determinant of prior covariance
        Returns: log determinant

        """
        log_det = 0.
        for K in self.Ks:
            rank_d = self.n / K.shape[0]
            det = np.linalg.slogdet(K)[1]
            log_det += rank_d * det
        return log_det

    def initialize_Rs(self):
        """
        Initializes upper triangular decomp of kronecker decomp of vairational covariance
        using identity matrix
        Returns: Rs (identity matrices)

        """
        return [np.eye(K.shape[0]) for K in self.Ks]

    def initialize_Rs_prior(self):
        """
        Initializes Rs using cholesky decomps of prior covariance
        Returns: Rs (cholesky decomp of prior covariances)

        """
        return [np.transpose(np.linalg.cholesky(K))
                for K in self.Ks]

    def full_S(self):
        """
        Returns full variaitonal covariance (based on kronecker decomps)
        Returns: Full variational covariance

        """
        return kron_list([R.T.dot(R) for R in self.Rs])

    def full_K(self):
        """

        Returns: full prior covariance

        """
        return kron_list(self.Ks)

    def predict(self):
        """
        GP predictions
        Returns: predictions

        """
        Ks = []
        for i in range(self.X.shape[1]):
            K = self.kernels[i].eval(self.kernels[i].params,
                                 np.expand_dims(np.unique(self.X[:, i]), 1))
            Ks.append(K)
        f_pred = kron_mvp(Ks, kron_mvp(self.K_invs, self.q_mu))
        return f_pred

    def sample_post(self):
        """
        Draws a sample from the GPR posterior
        Returns: sample

        """

        eps = np.random.normal(size = self.n)
        return self.q_mu + kron_mvp(self.Rs, eps)
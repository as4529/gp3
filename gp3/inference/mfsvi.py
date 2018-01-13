import autograd.numpy as np
from autograd import elementwise_grad as egrad, jacobian
from gp3.utils.structure import kron_mvp, kron_list_diag
from scipy.linalg import toeplitz
from scipy.linalg import solve
from gp3.utils.optimizers import AdamOptimizer
from tqdm import trange, tqdm_notebook

"""
Stochastic Variational Inference for Gaussian Processes with Non-Gaussian Likelihoods
"""

class MFSVI:


    def __init__(self, kernel_func, kernel_params, likelihood, X, y,
                 mu = None, obs_idx=None, max_grad = 1e1, noise = 1e-2,
                 step_size = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-1,
                 opt_kernel = False):
        """
        Args:
            kernel (GPy.Kernel): kernel function
            likelihood (): likelihood function. Requires log_like(), grad(), and hess()
            functions
            X (): data
            y (): responses
            mu (): prior mean
            noise (): noise variance
            obs_idx (): if dealing with partial grid, indices of grid that are observed
            verbose (): print or not
        """

        self.X = X
        self.y = y
        self.n, self.d = self.X.shape
        self.X_dims = [np.expand_dims(np.unique(X[:,i]), 1) for i in range(self.d)]
        if mu is None:
            self.mu = np.zeros(self.n)
        else:
            self.mu = mu
        self.obs_idx = obs_idx
        self.max_grad = max_grad
        self.init_Ks(kernel_func, kernel_params, noise, opt_kernel)
        self.elbos = []

        self.q_mu = self.mu
        self.q_S = np.ones(self.n)*np.log(self.Ks[0][0,0]**self.d)
        self.optimizer = AdamOptimizer(step_size, b1, b2, eps)

        self.likelihood = likelihood
        self.likelihood_opt = egrad(self.likelihood.log_like)


    def run(self, its, n_samples=1, notebook_mode = True):
        """
        Runs stochastic variational inference
        Args:
            its (int): Number of iterations
            n_samples (int): Number of samples for SVI
        Returns: Nothing, but updates instance variables
        """

        if notebook_mode == True:
            t = tqdm_notebook(xrange(its), leave=False)
        else:
            t = trange(its, leave = False)
        v_mu, v_s, v_k, m_mu, m_s, m_k  = (None for _ in range(6))

        for i in t:

            KL_grad_S, KL_grad_mu = self.grad_KL_S(), self.grad_KL_mu()
            grads_mu, grads_S, es, rs= ([] for i in range(4))

            for j in range(n_samples):
                eps = np.random.normal(size=self.n)
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
            kern_and_grad = None
            if self.opt_kernel == True:
                kern_grad = self.nat_grad_kern()
                kern_grad_clip = np.clip(kern_grad, -self.max_grad, self.max_grad)
                kern_and_grad = (self.kernel_params, kern_grad_clip)
                self.kern_grad = kern_grad

            self.q_mu, m_mu, v_mu = self.optimizer.step(mu_vars, m_mu, v_mu, i+1)
            self.q_S, m_s, v_s = self.optimizer.step(S_vars, m_s, v_s, i+1)
            if self.opt_kernel == True:
                self.kernel_params, m_k, v_k = self.optimizer.step(kern_and_grad,
                                                                   m_k, v_k, i+1)
                self.Ks, self.K_invs = self.construct_Ks()

            if i > 100 and self.loss_check() == True:
                print "converged at", i, "iterations"
                return

        return

    def eval_obj(self, S, q_mu, rs, kern_params = None):
        """
        Evaluates variational objective
        Args:
            Rs (): Variational covariances (Cholesky decomposition of Kronecker decomp)
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
            Rs (): Variational covariance
            q_mu (): Variational mean
        Returns: KL divergence between q and p
        """

        if kern_params is None:
            K_invs = self.K_invs
            k_inv_diag = self.k_inv_diag
            det_K = self.det_K
        else:
            Ks, K_invs = self.construct_Ks(kernel_params = kern_params)
            k_inv_diag = kron_list_diag(K_invs)
            det_K = self.log_det_K(Ks)

        k_inv_mu = kron_mvp(K_invs, self.mu - q_mu)
        mu_penalty = np.sum(np.multiply(self.mu -q_mu, k_inv_mu))
        det_S = np.sum(S)
        trace_term = np.sum(np.multiply(k_inv_diag, np.exp(S)))

        kl = 0.5 * (det_K - self.n - det_S +
                    trace_term + mu_penalty)

        return max(0, kl)

    def grad_KL_S(self):
        """
        Gradient of KL divergence w.r.t variational covariance
        Returns: returns gradient
        """
        euc_grad = 0.5 * (-1. + np.multiply(self.k_inv_diag, np.exp(self.q_S)))
        nat_adj = 2. / (np.exp(-self.q_S) * np.square(self.q_mu))

        return np.multiply(nat_adj, euc_grad)

    def grad_KL_mu(self):
        """
        Gradient of KL divergence w.r.t variational mean
        Returns: returns gradient
        """
        return np.multiply(np.exp(self.q_S), -kron_mvp(self.K_invs, self.mu - self.q_mu))

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

        if self.obs_idx is not None:
            grad_mu = np.zeros(self.n)
            grad_mu[self.obs_idx] = dr
        else:
            grad_mu = dr
        grad_S = np.multiply(grad_mu, np.multiply(eps,
                                      np.multiply(0.5/np.sqrt(np.exp(self.q_S)),
                                                  np.exp(self.q_S))))

        return grad_S, grad_mu

    def grad_kern(self):
        """

        Returns: Gradient of KL w.r.t base kernel parameters

        """

        k_inv_mu =  self.q_mu - self.mu
        grads = []
        term1 = np.ones(len(self.kernel_params))
        term2 = np.expand_dims(np.ones(len(self.kernel_params)), 1)

        for i, X in enumerate(reversed(self.X_dims)):

            grad = self.kernel_opt(self.kernel_params, X[0], X)
            toep_grad = np.stack([toeplitz(grad[:, :, k]) for k in xrange(grad.shape[2])],
                                 axis = -1)
            grads.append(toep_grad)

            term1 = np.multiply(term1,
                                np.einsum('ij, ijk -> k', self.K_invs[i], toep_grad))

            diag = np.einsum('ij,jik->jk', self.K_invs[i],
                np.einsum('ijk,j...',toep_grad, self.K_invs[i]))

            term2 = np.stack([np.hstack([ii * term2[d,:]
                              for ii in diag[:,d]]) for d in range(diag.shape[1])])

            k_inv_mu = np.reshape(k_inv_mu, [X.shape[0], -1])
            k_inv_mu = np.dot(self.K_invs[i], k_inv_mu).T

        term2 = np.sum(np.multiply(term2.T, np.expand_dims(np.exp(self.q_S), 1)), 0)
        k_inv_mu = np.reshape(k_inv_mu, [-1])
        term3 = np.tile(k_inv_mu, (len(self.kernel_params), 1)).T

        for grad in grads:
            term3 = np.reshape(term3, [grad.shape[0], -1, grad.shape[2]])
            term3 = np.stack([np.dot(grad[:,:,d], term3[:,:,d]).T for
                              d in range(len(self.kernel_params))], axis = -1)

        term3 = np.reshape(term3, [-1, len(self.kernel_params)])
        term3 = np.sum(np.multiply(np.expand_dims(k_inv_mu,1), term3), 0)

        return -0.5*(term1 - term2 - term3), grads

    def nat_grad_kern(self):
        """

        Returns: Natural gradient of KL w.r.t base kernel parameters

        """
        euc_grad, grads = self.grad_kern()
        fisher = np.zeros((len(self.kernel_params), len(self.kernel_params)))

        for i in range(len(self.kernel_params)):
            for j in range(len(self.kernel_params)):
                fisher[i,j] = 0.5*kron_list_diag([np.dot(self.K_invs[d], grads[d][:,:,i]).
                                                 dot(self.K_invs[d]).dot(grads[d][:,:,j])
                                                 for d in range(self.d)]).sum()
        nat_grad = solve(fisher, euc_grad)

        return nat_grad

    def construct_Ks(self, kernel=None, kernel_params = None):
        """
        Constructs kronecker-decomposed kernel matrix
        Args:
            kernel (): kernel (if not using kernel passed in constructor)
        Returns: Rist of kernel evaluated at each dimension
        """

        if kernel is None:
            kernel = self.kernel_func
        if kernel_params is None:
            kernel_params = self.kernel_params

        Ks = [kernel(kernel_params, X_dim) +
              np.diag(np.ones(X_dim.shape[0])) * self.noise for X_dim in self.X_dims]
        K_invs = [np.linalg.inv(K) for K in Ks]

        return Ks, K_invs

    def init_Ks(self, kernel_func, kernel_params, noise, opt_kernel):

        self.kernel_func, self.kernel_params = kernel_func, kernel_params
        self.noise = noise
        self.Ks, self.K_invs = self.construct_Ks()
        self.k_inv_diag = kron_list_diag(self.K_invs)
        self.det_K = self.log_det_K()
        self.opt_kernel = opt_kernel
        if opt_kernel == True:
            self.kernel_opt = jacobian(self.kernel_func)

    def log_det_K(self, Ks = None):
        """
        Log determinant of prior covariance
        Returns: log determinant
        """
        if Ks is None:
            Ks = self.Ks

        log_det = 0.

        for K in Ks:
            rank_d = self.n / K.shape[0]
            det = np.linalg.slogdet(K)[1]
            log_det += rank_d * det

        return log_det

    def predict(self):
        """
        GP predictions
        Returns: predictions
        """
        Ks = [self.kernel_func(self.kernel_params, X_dim) for X_dim in self.X_dims]

        f_pred = kron_mvp(Ks, kron_mvp(self.K_invs, self.q_mu))

        return f_pred

    def sample_post(self, n_samples = 1):

        return self.q_mu + \
               np.multiply(np.expand_dims(np.sqrt(np.exp(self.q_S)), 1),
                           np.random.normal(size = (self.n, n_samples))).flatten()

    def loss_check(self):

        return sum(x >= y for x, y in zip(self.elbos[-100:], self.elbos[-99:])) > 50

    def line_search(self, S_grads, mu_grads, kern_grads,
                    obj_init, es, min_step = 1e-9):
        """
        Performs line search to find optimal step size
        Args:
            Rs_grads (): Gradients of R (variational covariances)
            mu_grads (): Gradients of mu (variational mean)
            obj_init (): Initial objective value
            r (): transformed random Gaussian sample
            eps (): random Gaussian sample
            min_step (): minimum step for backtracking line search
        Returns: Optimal step size
        """
        step = 1.
        t = 0
        while step > min_step:

            S_search = S_grads[0] + step * S_grads[1]
            mu_search = mu_grads[0] + step * mu_grads[1]

            if kern_grads is not None:
                kern_search = kern_grads[0] + step * kern_grads[1]
            else:
                kern_search = None

            rs_search = [mu_search + np.multiply(np.sqrt(np.exp(S_search)), eps)
                         for eps in es]

            obj_search, kl_search, like_search = self.eval_obj(S_search, mu_search,
                                                               rs_search,
                                                               kern_search)

            if obj_init - obj_search > step*t:
                self.q_S, self.q_mu  = S_search, mu_search
                if kern_grads is not None:
                    self.kernel_params = kern_search
                    self.Ks, self.K_invs = self.construct_Ks()
                return

            step = step * 0.5
            t += 1

        return None
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from gp3.utils.kron import kron_list, kron_mvp
from tqdm import trange

class FullSVI:


    def __init__(self, kernel, likelihood, X, y, mu, noise = 1e-2, obs_idx=None,
                 verbose = False):
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
        self.d = self.X.shape[1]
        self.y = y
        self.n = self.X.shape[0]
        self.noise = noise
        self.likelihood = likelihood
        self.kernel = kernel
        self.elbos = []
        self.likes = []
        self.kls = []
        self.obs_idx = obs_idx
        self.verbose = verbose
        self.grad_Rs = []
        self.grad_mus = []

        self.Ks, self.K_invs = self.construct_Ks()
        self.det_K = self.log_det_K()
        self.Rs = self.initialize_Rs_prior()
        self.mu = mu
        self.trace_term, self.traces = self.calc_trace_term()
        self.cg_opt = CGOptimizer(self.cg_prod)

        self.likelihood_opt = egrad(self.likelihood.log_like)
        self.q_mu = self.mu

    def run(self, its):
        """
        Runs stochastic variational inference
        Args:
            its (): Number of iterations
        Returns: Nothing, but updates instance variables
        """

        t = trange(its, leave=True)

        for i in t:
            self.trace_term, self.traces = self.calc_trace_term()
            KL_grad_R = self.grad_KL_R()
            KL_grad_mu = self.grad_KL_mu()

            eps = np.random.normal(size = self.n)
            r = self.q_mu + kron_mvp(self.Rs, eps)
            like_grad_R, like_grad_mu = self.grad_like(r, eps)

            grad_R = [np.clip(-KL_grad_R[i] + like_grad_R[i], -1e3, 1e3)
                      for i in range(len(KL_grad_R))]
            grad_mu= np.clip(-KL_grad_mu + like_grad_mu, -1e3, 1e3)
            R_and_grads = zip(grad_R, self.Rs)
            mu_and_grad = (grad_mu, self.q_mu)

            obj, kl, like = self.eval_obj(self.Rs, self.q_mu, r)
            self.elbos.append(-obj)
            self.kls.append(kl)
            self.likes.append(like)

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

        while step > 1e-9:

            R_search = [R + step*R_grad
                        for R_grad, R in Rs_grads]
            mu_search = mu_grads[1] + step*mu_grads[0]
            r_search = mu_search + kron_mvp(R_search, eps)
            obj_search, kl_search, like_search = self.eval_obj(R_search, mu_search,
                                                               r_search)
            if obj_init - obj_search > step:
                return R_search, mu_search, obj_search, step

            step = step*0.5

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
        mu_penalty = np.sum(np.multiply(self.mu -q_mu, k_inv_mu))
        det_S = self.log_det_S(Rs)
        trace_term = self.calc_trace_term(Rs)[0]
        kl = 0.5 * (self.det_K - self.n - det_S +
                      trace_term + mu_penalty)

        if kl < 0:
            return 0.

        return max(0, kl)

    def grad_KL_R(self):
        """
        Gradient of KL divergence w.r.t variational covariance
        Returns: returns gradient
        """
        return [np.diag(-2*self.n/self.Rs[d].shape[0]/
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
            grad_mu = np.zeros(self.n)
            grad_mu[self.obs_idx] = dr
        else:
            grad_mu = dr

        return grads_R, grad_mu

    def construct_Ks(self, kernel=None):
        """
        Constructs kronecker-decomposed kernel matrix
        Args:
            kernel (): kernel (if not using kernel passed in constructor)
        Returns: Rist of kernel evaluated at each dimension
        """

        if kernel is None:
            kernel = self.kernel

        Ks = []
        for i in range(self.X.shape[1]):
            K = kernel.K(np.expand_dims(np.unique(self.X[:, i]), 1))
            K = K + np.diag(np.ones(K.shape[0]))*self.noise
            Ks.append(K)

        K_invs = [np.linalg.inv(K) for K in Ks]

        return Ks, K_invs

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

        return 2*np.sum([self.n/R.shape[0]*
                                 np.sum(np.log(np.diag(R)))
                                 for R in Rs])

    def log_det_K(self):
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
        return[np.eye(K.shape[0])
            for K in self.Ks]

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
        return kron_list([np.dot(np.transpose(R), R) for R in self.Rs])


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
            K = self.kernel_func(self.kernel_params,
                                 np.expand_dims(np.unique(self.X[:, i]), 1))
            Ks.append(K)

        f_pred = kron_mvp(Ks, kron_mvp(self.K_invs, self.q_mu))

        return f_pred

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

        eps = np.random.normal(size = self.n)
        r = self.q_mu + kron_mvp(self.Rs, eps)

        return r
import autograd.numpy as np
from autograd import elementwise_grad as egrad, jacobian
from gp3.utils.structure import kron_mvp, kron_list_diag

class InfBase(object):

    def __init__(self, X, y, kernel, likelihood = None, mu = None, obs_idx = None,
                 opt_kernel = False, max_grad = 1e2, noise = 1e-2):

        self.X = X
        self.y = y
        self.n, self.d = self.X.shape
        self.X_dims = [np.expand_dims(np.unique(X[:, i]), 1) for i in range(self.d)]
        if mu is None:
            self.mu = np.zeros(self.n)
        else:
            self.mu = mu
        self.obs_idx = obs_idx
        self.max_grad = max_grad
        self.init_Ks(kernel, noise, opt_kernel)

        if likelihood is not None:
            self.likelihood = likelihood
            self.likelihood_opt = egrad(self.likelihood.log_like)

    def init_Ks(self, kernel, noise, opt_kernel):

        self.kernel = kernel
        self.noise = noise
        self.Ks, self.K_invs = self.construct_Ks()
        self.k_inv_diag = kron_list_diag(self.K_invs)
        self.det_K = self.log_det_K()
        self.K_eigs = [np.linalg.eig(K) for K in self.Ks]
        self.opt_kernel = opt_kernel
        if opt_kernel == True:
            self.kernel_opt = jacobian(self.kernel.eval)

    def log_det_K(self, Ks=None):
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

    def construct_Ks(self, kernel=None):
        """
        Constructs kronecker-decomposed kernel matrix
        Args:
            kernel (): kernel (if not using kernel passed in constructor)
        Returns: Rist of kernel evaluated at each dimension
        """

        if kernel is None:
            kernel = self.kernel

        Ks = [kernel.eval(kernel.params, X_dim) +
              np.diag(np.ones(X_dim.shape[0])) * self.noise for X_dim in self.X_dims]
        K_invs = [np.linalg.inv(K) for K in Ks]

        return Ks, K_invs
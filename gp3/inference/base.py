import autograd.numpy as np
from autograd import elementwise_grad as egrad, jacobian
from gp3.utils.structure import kron_list, kron_list_diag

class InfBase(object):

    def __init__(self,
                 X,
                 y,
                 kernels,
                 likelihood=None,
                 mu=None,
                 obs_idx=None,
                 max_grad=10.,
                 noise=1e-6):

        self.X = X
        self.y = y
        self.m = self.X.shape[0]
        self.d = self.X.shape[1]
        self.obs_idx = obs_idx
        self.n = len(self.obs_idx) if self.obs_idx is not None else self.m
        self.X_dims = [np.expand_dims(np.unique(X[:, i]), 1) for i in range(self.d)]
        self.mu = np.zeros(self.m) if mu is None else mu
        self.max_grad = max_grad
        self.init_Ks(kernels, noise)
        if likelihood is not None:
            self.likelihood = likelihood
            self.likelihood_grad = egrad(self.likelihood.log_like)

    def init_Ks(self, kernels, noise):

        self.kernels = kernels
        self.noise = np.array([noise])
        self.construct_Ks()

    def construct_Ks(self, kernels=None):
        """
        Constructs kronecker-decomposed kernel matrix
        Args:
            kernel (): kernel (if not using kernel passed in constructor)
        Returns: Rist of kernel evaluated at each dimension
        """
        kernels = self.kernels if kernels is None else kernels
        self.Ks = [kernels[d].eval(kernels[d].params, self.X_dims[d])
              for d in range(self.d)]
        self.K_eigs = [np.linalg.eig(K) for K in self.Ks]
        self.det_K = self.log_det_K()
        return

    def log_det_K(self, Ks=None):
        """
        Log determinant of prior covariance
        Returns: log determinant
        """
        Ks = self.Ks if Ks is None else Ks
        log_det = 0.
        for K in Ks:
            rank_d = self.m / K.shape[0]
            det = np.linalg.slogdet(K)[1]
            log_det += rank_d * det
        return log_det

    def full_K(self):

        return kron_list(self.Ks)

    def full_A(self):

        return self.full_K() + np.diag(np.ones(self.m) * self.noise)
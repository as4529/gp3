
"""
Vanilla GP inference with kronecker structure
"""
import autograd.numpy as np
from autograd import elementwise_grad as egrad, jacobian
from gp3.utils.structure import kron_mvp, kron_list_diag
from tqdm import trange
from scipy.linalg import toeplitz
from scipy.linalg import solve
from gp3.kernels.kernels import softmax


class Vanilla:


    def __init__(self, kernel_func, kernel_params, X, y,
                 mu = None, obs_idx=None, opt_kernel = False):
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

        self.objs = []
        self.grad_norms = []

        self.kernel_func, self.kernel_params = kernel_func, kernel_params
        self.Ks, self.K_invs = self.construct_Ks()


    def marginal(self):

        k_inv_y = kron_mvp(self.K_invs, self.y)


        return
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

        Ks = [kernel(kernel_params, X_dim) for X_dim in self.X_dims]
        K_invs = [np.linalg.inv(K) for K in Ks]

        return Ks, K_invs
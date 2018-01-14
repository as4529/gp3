import autograd.numpy as np
from gp3.utils.transforms import softplus

class RBF:

    def __init__(self, params = None):

        self.params = params

    def eval(self, params, X, X2 = None):
        """
        RBF (squared exponential) kernel with softplus transform of lengthscale and variance
        Args:
            params (np.array): lengthscale and variance (in order)
            X (): first X
            X2 (): second X (if there is one, otherwise just eval on X)

        Returns:s

        """

        ls, var = self.unpack_params(params)

        if X2 is None:
            X2 = X

        delta = np.expand_dims(X / ls, 1) -\
                np.expand_dims(X2 / ls, 0)

        return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2))

    def unpack_params(self, params):

        return softplus(params[0]), softplus(params[1])

class DeepRBF:

    def __init__(self, params = None):

        self.params = params

    def eval(self, params, X, X2 = None):
        """
        RBF (squared exponential) kernel with softplus transform of lengthscale and variance
        Args:
            params (np.array): lengthscale and variance (in order)
            X (): first X
            X2 (): second X (if there is one, otherwise just eval on X)

        Returns:s

        """

        ls, var, weights = self.unpack_params(params)

        if X2 is None:
            X2 = X

        delta = np.expand_dims(X / ls, 1) -\
                np.expand_dims(X2 / ls, 0)

        return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2))

    def unpack_params(self, params):

        return softplus(params[0]), softplus(params[1]), params[2:]

    def nn_predict(self, weights, X):

        return

class SpectralMixture:

    def __init__(self, params):
        self.params = params

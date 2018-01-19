import autograd.numpy as np
from gp3.utils.transforms import softplus, inv_softplus

class RBF:

    def __init__(self, lengthscale, variance, noise):

        self.params = self.pack_params(lengthscale, variance, noise)

    def eval(self, params, X, X2 = None):
        """
        RBF (squared exponential) kernel with softplus transform of lengthscale and variance
        Args:
            params (np.array): lengthscale and variance (in order)
            X (): first X
            X2 (): second X (if there is one, otherwise just eval on X)

        Returns:s

        """

        ls, var, noise = self.unpack_params(params)

        if X2 is None:
            X2 = X

        delta = np.expand_dims(X / ls, 1) -\
                np.expand_dims(X2 / ls, 0)

        return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2)) +\
               np.diag(np.ones(X.shape[0]))*noise

    def unpack_params(self, params):

        return softplus(params[0]), softplus(params[1]), softplus(params[2])

    def pack_params(self, lengthscale, variance, noise):

        return inv_softplus(np.array([lengthscale, variance, noise]))

class DeepRBF:
    """
    Inspired by autograd's Bayesian neural net example
    """

    def __init__(self, lengthscale, variance, noise,
                 layer_sizes):

        shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        num_weights = sum((m + 1) * n for m, n in shapes)
        weights = np.random.normal(size = num_weights)

        self.params = self.pack_params(lengthscale, variance, noise, weights)
        self.shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))

    def eval(self, params, X, X2 = None):
        """
        RBF (squared exponential) kernel with softplus
         transform of lengthscale and variance
        Args:
            params (np.array): lengthscale and variance (in order)
            X (): first X
            X2 (): second X (if there is one, otherwise just eval on X)

        Returns:s

        """

        ls, var, noise, weights = self.unpack_params(params)
        X_nn = self.nn_predict(weights, X)

        if X2 is not None:
            X2_nn = self.nn_predict(weights, X2)
        else:
            X2_nn = X_nn

        delta = np.expand_dims(X_nn / ls, 1) -\
                np.expand_dims(X2_nn / ls, 0)

        return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2)) +\
               np.diag(np.ones(X.shape[0]))*noise

    def unpack_params(self, params):

        return softplus(params[0]), softplus(params[1]), \
               softplus(params[1]), params[2:]

    def unpack_layers(self, weights):

        for m, n in self.shapes:
            yield weights[:m * n].reshape((m, n)), \
                  weights[m * n:m * n + n].reshape((1, n))
            weights = weights[(m + 1) * n:]

    def nn_predict(self, weights, inputs):

        outputs = None
        for W, b in self.unpack_layers(weights):
            outputs = np.dot(inputs, W) + b
            inputs = softplus(outputs)

        return outputs

    def pack_params(self, lengthscale, variance, noise, weights):

        return np.hstack([inv_softplus(np.array([lengthscale, variance, noise])),
                         weights])

class SpectralMixture:

    def __init__(self, params):
        self.params = params

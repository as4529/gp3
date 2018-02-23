import autograd.numpy as np
from gp3.utils.transforms import softplus, inv_softplus

class RBF:

    def __init__(self, lengthscale, variance):

        self.lengthscale = lengthscale
        self.variance = variance
        self.params = self.pack_params(lengthscale, variance)

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

    def pack_params(self, lengthscale, variance):

        return inv_softplus(np.array([lengthscale, variance]))

class DeepRBF:
    """
    RBF kernel after transformation of inputs with a feed-forward NN.
    Inspired by autograd's Bayesian neural net example.
    """

    def __init__(self, lengthscale, variance, noise,
                 layer_sizes, penalty = 1e-2):

        shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        num_weights = sum((m + 1) * n for m, n in shapes)
        weights = np.random.normal(size = num_weights)

        self.params = self.pack_params(lengthscale, variance, noise, weights)
        self.shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.penalty = penalty

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

    def penalty(self):

        return np.linalg.norm(self.params) * self.penalty

class Matern52:

    def __init__(self, lengthscale, variance, penalty = 1e-2):

        self.lengthscale = lengthscale
        self.variance = variance
        self.params = self.pack_params(lengthscale, variance)

    def eval(self, params, X, X2=None):
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

        delta = np.expand_dims(X / ls, 1) - \
                np.expand_dims(X2 / ls, 0)

        d2 = np.sum(delta ** 2, axis=2)
        d = np.sqrt(d2)

        self.d = d
        self.d2 = d2

        return var * (1 + np.sqrt(5)* d / ls + 5 * d2
                / (3 * np.square(ls))) * \
                np.exp( - np.sqrt(5) * d / ls)

    def unpack_params(self, params):
        return softplus(params[0]), softplus(params[1])

    def pack_params(self, lengthscale, variance):
        return inv_softplus(np.array([lengthscale, variance]))

class SpectralMixture:

    def __init__(self, params):
        self.params = params

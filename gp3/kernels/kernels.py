import autograd.numpy as np
from gp3.utils.transforms import softplus, inv_softplus,\
    softmax, unit_norm

class RBF:

    def __init__(self, lengthscale, variance, l=0., prior_ls=10.,
                 prior_var=1.):

        self.lengthscale = lengthscale
        self.variance = variance
        self.prior_ls = prior_ls
        self.prior_var = prior_var
        self.l = l
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
        X2 = X if X2 is None else X2
        delta = np.expand_dims(X / ls, 1) -\
                np.expand_dims(X2 / ls, 0)
        return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2))

    def unpack_params(self, params):

        return softplus(params[0]), softplus(params[1])

    def pack_params(self, lengthscale, variance):

        return inv_softplus(np.array([lengthscale, variance]))

    def log_prior(self, params):
        ls, var = self.unpack_params(params)
        ls_pen = self.l * np.linalg.norm(ls - self.prior_ls)
        var_pen = self.l * np.linalg.norm(var - self.prior_var)
        return ls_pen + var_pen

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
        X2_nn = self.nn_predict(weights, X2) if X2 is not None else X_nn
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
        X2 = X if X2 is None else X2
        delta = np.expand_dims(X / ls, 1) - \
                np.expand_dims(X2 / ls, 0)
        d2 = np.sum(delta ** 2, axis=2)
        d = np.sqrt(d2)
        return var * (1 + np.sqrt(5)* d / ls + 5 * d2
                / (3 * np.square(ls))) * \
                np.exp( - np.sqrt(5) * d / ls)

    def unpack_params(self, params):
        return softplus(params[0]), softplus(params[1])

    def pack_params(self, lengthscale, variance):
        return inv_softplus(np.array([lengthscale, variance]))

class SpectralMixture:

    def __init__(self, w, mu, sigma, l=0.):
        self.w = w
        self.mu = mu
        self.sigma = sigma
        self.a = len(w)
        self.l = l
        self.params = self.pack_params(w, mu, sigma)

    def eval(self, params, X, X2 = None):

        w, mu, sigma = self.unpack_params(params)
        X2 = X if X2 is None else X2
        delta = np.expand_dims(X, 1) - \
                np.expand_dims(X2, 0)
        tau_sq = np.sum(delta ** 2, axis=2)
        tau = np.sqrt(tau_sq)
        out = np.zeros((X.shape[0], X2.shape[0]))
        for a in range(self.a):
            out += w[a] * np.exp(-2 * np.pi * tau_sq * sigma[0]) * \
                    np.cos(2 * np.pi * tau * mu[a])
        return out

    def unpack_params(self, params):
        return softplus(params[:self.a]), softplus(params[self.a: 2 * self.a]),\
               softplus(params[2 * self.a: len(params)])

    def pack_params(self, w, mu, sigma):
        return inv_softplus(np.hstack([w, mu, sigma]))

    def log_prior(self, params):
        return self.l * np.linalg.norm(params)

class Task:

    def __init__(self, L, n_tasks, l=100):
        self.L = L
        self.l = l
        self.n_tasks = n_tasks

    def eval(self, params, X, X2=None):
        K = self.unpack_params(params)
        return K

    def unpack_params(self, params):
        tri = np.zeros((self.n_tasks, self.n_tasks))
        tri[np.triu_indices(self.n_tasks, 1)] = params
        return tri.dot(tri.T)

    def pack_params(self, L):
        packed = L.flatten()
        return packed

    def log_prior(self, params):
        return self.l * np.linalg.norm(params)


class TaskEmbed:

    def __init__(self, Q, n_tasks, n_dims, l=100.):
        self.Q = Q
        self.n_tasks = n_tasks
        self.n_dims = n_dims
        self.l = l
        self.params = self.pack_params(self.Q)

    def eval(self, params, X, X2 = None):
        Q = self.unpack_params(params)
        return np.dot(Q, Q.T)

    def unpack_params(self, params):
        return np.reshape(params,
                          (self.n_tasks, self.n_dims))

    def pack_params(self, Q):
        packed = Q.flatten()
        return packed

    def log_prior(self, params):
        return self.l * np.linalg.norm(params)


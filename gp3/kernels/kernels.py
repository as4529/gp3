import autograd.numpy as np

def rbf(params, X, X2 = None):
    """
    RBF (squared exponential) kernel with softplus transform of lengthscale and variance
    Args:
        params (np.array): lengthscale and variance (in order)
        X (): first X
        X2 (): second X (if there is one, otherwise just eval on X)

    Returns:

    """

    ls, var = softmax(params[0]), softmax(params[1])

    if X2 is None:
        X2 = X

    delta = np.expand_dims(X / ls, 1) -\
            np.expand_dims(X2 / ls, 0)

    return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2))

def deep_rbf(params, X, X2 = None):
    """
    RBF (squared exponential) kernel with softplus transform of lengthscale and variance
    Args:
        params (np.array): lengthscale and variance (in order)
        X (): first X
        X2 (): second X (if there is one, otherwise just eval on X)

    Returns:

    """

    ls, var = softmax(params[0]), softmax(params[1])

    if X2 is None:
        X2 = X

    delta = np.expand_dims(X / ls, 1) -\
            np.expand_dims(X2 / ls, 0)

    return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2))

def spectral_mixture(params, X, X2 = None):

    return

def softmax(x):

    return np.log(np.exp(x) + 1)

def inv_softmax(x):

    return np.log(np.exp(x) - 1)
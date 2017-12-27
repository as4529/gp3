import autograd.numpy as np

def rbf(params, X, X2 = None):

    lengthscale, variance = params

    X = X / lengthscale

    if X2 is None:
        X2 = X
    else:
        X2 = X2 / lengthscale

    delta = np.expand_dims(X / lengthscale, 1) -\
            np.expand_dims(X2 / lengthscale, 0)

    return variance * np.exp(-0.5 * np.sum(delta ** 2, axis=2))


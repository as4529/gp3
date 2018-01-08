import autograd.numpy as np

def rbf(params, X, X2 = None):

    ls, var = np.log(np.exp(params[0]) + 1), np.log(np.exp(params[1] + 1))

    if X2 is None:
        X2 = X

    delta = np.expand_dims(X / ls, 1) -\
            np.expand_dims(X2 / ls, 0)

    return var * np.exp(-0.5 * np.sum(delta ** 2, axis=2))


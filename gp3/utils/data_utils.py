import GPy
import numpy as np
import itertools

"""
basic utilities for simulating data
"""


def sim_f(X, k=GPy.kern.RBF(input_dim=2, variance=1., lengthscale=10), mu=3):
    """
    simulates function values given X
    Args:
        X (np.array): data points
        k (GPy.kernel): kernel function
        mu (np.array): prior mean

    Returns: sampled function values

    """
    return np.random.multivariate_normal(np.ones(X.shape[0]) * mu, k.K(X, X) +
                                         np.diag(np.ones(X.shape[0]))*1e-12)

def sim_X(D=2, N_dim=30, lower=0, upper=100):
    """
    Simulates X on a rectilinear grid
    Args:
        D (int): dimensions
        N_dim (int): number of points per dimension
        lower (float): lower bound for uniform draw
        upper (float): upper bound for uniform draw

    Returns: points on a grid

    """
    grid = [np.sort(np.random.uniform(lower, upper, size=N_dim))
            for d in range(D)]

    return np.array(list(itertools.product(*grid)))


def sim_X_equispaced(D=2, N_dim=20, lower=0, upper=100):

    grid = [np.arange(lower, upper, (upper-lower)*1.0/N_dim) for d in range(D)]

    return np.array(list(itertools.product(*grid)))


def poisson_draw(f, noise_val):
    """

    Args:
        f (np.array): draws a poisson based on function values
                      (with some noise added)
        noise_val(float): between zero and one, add normal noise to f

    Returns: poisson draws

    """
    return np.random.poisson(np.exp(f + noise_val * np.random.normal(0, 1,
                             size=len(f))))


def rand_partial_grid(X, y, prop):

    indices = np.sort(np.random.choice(X.shape[0], int(X.shape[0] * prop),
                      replace=False))
    X_partial = X[indices]
    y_partial = y[indices]
    X_partial = X_partial[np.lexsort((X_partial[:, 1], X_partial[:, 0]))]

    return X_partial, y_partial

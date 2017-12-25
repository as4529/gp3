import numpy as np
import itertools
import tensorflow as tf
from scipy.stats import rankdata

def fill_grid(X, y):
    """
    Fills a partial grid with "imaginary" observations
    Args:
        X (np.array): data that lies on a partial grid

    Returns:
        X_grid: full grid X (including real and imagined points)
        y_full: full grid y (with zeros corresponding to imagined points)
        obs_idx: indices of observed points
        imag_idx: indices of imagined points

    """

    D = X.shape[1]
    x_dims = [np.unique(X[:, d]) for d in range(D)]

    X_grid = np.array(list(itertools.product(*x_dims)))

    d_indices = [{k: v for k, v in zip(x_dims[d], range(x_dims[d].shape[0]))}
                 for d in range(D)]
    grid_part = np.ones([x_d.shape[0] for x_d in x_dims])*-1

    for i in range(X.shape[0]):
        idx = tuple([d_indices[d][X[i, d]] for d in range(D)])
        grid_part[idx] = 1

    obs_idx = np.where(grid_part.flatten() > -1)[0]
    imag_idx = np.where(grid_part.flatten() == -1)[0]

    y_full = np.zeros(X_grid.shape[0])
    y_full[obs_idx] = y

    return X_grid, y_full, obs_idx, imag_idx

def weights_nn(X, U):

    dist = np.reshape(np.sum(np.square(X), 1), [-1, 1]) + np.reshape(
        np.sum(np.square(U), 1), [1, -1]) - 2 * np.dot(X, U.T)
    inv_dist = 1.0/dist
    ranks = np.vstack([rankdata(i, method='ordinal') for i in inv_dist])
    idx = ranks > ranks.shape[1] - 4
    weights = inv_dist * idx
    weights_norm = weights / weights.sum(1, keepdims=1)

    return weights_norm



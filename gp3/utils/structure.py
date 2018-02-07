import numpy as np
from functools import reduce

def kron(A, B):
    """
    Kronecker product of two matrices
    Args:
        A (np.array): first matrix for kronecker product
        B (np.array): second matrix

    Returns: kronecker product of A and B

    """

    n_col = A.shape[1] * B.shape[1]
    out = np.zeros([0, n_col])

    for i in range(A.shape[0]):

        row = np.zeros([B.shape[0], 0])

        for j in range(A.shape[1]):
            row = np.concatenate([row, A[i, j]* B], 1)

        out = np.concatenate([out, row], 0)

    return out


def kron_list(matrices):
    """
    Kronecker product of a list of matrices
    Args:
        matrices (list of np.array): list of matrices

    Returns:

    """
    return reduce(kron, matrices)


def kron_mvp(Ks, v):
    """
    Matrix vector product using Kronecker structure
    Args:
        Ks (list of np.array): list of matrices
        of K
        v (np.array): vector to multiply K by

    Returns: matrix vector product of K and v

    """

    mvp = v

    for k in reversed(Ks):
        mvp = np.reshape(mvp, [k.shape[0], -1])
        mvp = np.dot(k, mvp).T

    return mvp.flatten()

def kron_list_diag(Ks):

    diag = np.hstack([ii * np.diag(Ks[len(Ks)-1])
                      for ii in np.diag(Ks[len(Ks)-2])])

    for i in reversed(range(len(Ks[:-2]))):
        diag = np.hstack([ii * diag for ii in np.diag(Ks[i])])

    return diag


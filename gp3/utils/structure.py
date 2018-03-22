import numpy as np
from functools import reduce
from scipy.linalg import circulant
from copy import copy

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
    Kronecker matrix vector product
    Args:
        Ks (): kronecker decomposition of K
        v (): vector for product

    Returns: Ks*v

    """

    m = [k.shape[0] for k in Ks]
    n = [k.shape[1] for k in Ks]
    b = copy(v)

    for i in range(len(Ks)):
        a = np.reshape(b, (np.prod(m[:i], dtype=int),
                           n[i],
                           np.prod(n[i+1:], dtype=int)))
        tmp = np.reshape(np.swapaxes(a, 2, 1),
                         (-1, n[i]))
        tmp = tmp.dot(Ks[i].T)
        b = np.swapaxes(np.reshape(tmp,
                                   (a.shape[0],
                                    a.shape[2],
                                    m[i])), 2, 1)
    return np.reshape(b, np.prod(m, dtype=int))


def kron_list_diag(Ks):
    """
    Computes diagonal of Kronecker product
    Args:
        Ks (): Kronecker decomposition

    Returns:

    """


    diag = np.hstack([ii * np.diag(Ks[len(Ks)-1])
                      for ii in np.diag(Ks[len(Ks)-2])])
    for i in reversed(range(len(Ks[:-2]))):
        diag = np.hstack([ii * diag for ii in np.diag(Ks[i])])
    return diag

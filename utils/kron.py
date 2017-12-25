import numpy as np

def kron(A, B):
    """
    Kronecker product of two matrices
    Args:
        A (np.Variable): first matrix for kronecker product
        B (np.Variable): second matrix

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
        matrices (list of np.Variable): list of matrices

    Returns:

    """
    out = kron(matrices[0], matrices[1])

    for i in range(2, len(matrices)):
        out = kron(out, matrices[i])

    return out


def kron_mvp(Ks, v):
    """
    Matrix vector product using Kronecker structure
    Args:
        Ks (list of np.Variable): list of matrices
        of K
        v (np.Variable): vector to multiply K by

    Returns: matrix vector product of K and v

    """

    mvp = np.reshape(v,[Ks[-1].shape[0], -1])

    for idx, k in enumerate(reversed(Ks)):
        if idx > 0:
            rows = k.shape[0]
            mvp = np.reshape(mvp, [rows, -1])
        mvp = np.dot(k, mvp).T

    return np.reshape(mvp, [-1])

import numpy as np


class CGOptimizer:

    def __init__(self, cg_prod=None, tol=1e-3):
        self.cg_prod = cg_prod
        self.tol = tol

    def cg(self, A, b, x = None):

        n = len(b)
        if not x:
            x = np.ones(n)

        r = self.cg_prod(A, x) - b
        p = - r
        r_k_norm = np.dot(r, r)
        for i in xrange(2 * n):
            Ap = self.cg_prod(A, p)
            alpha = r_k_norm / np.dot(p, Ap)
            x += alpha * p
            r += alpha * Ap
            r_kplus1_norm = np.dot(r, r)
            beta = r_kplus1_norm / r_k_norm
            r_k_norm = r_kplus1_norm
            if r_kplus1_norm < self.tol:
                break
            p = beta * p - r

        return x
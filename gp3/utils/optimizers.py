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


class AdamOptimizer:

    def __init__(self, step_size, b1, b2, eps):
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self, var_and_grad, m, v, t):
        """
        Adapted from autograd.misc.optimizers
        Args:
            S_grads ():
            mu_grads ():
            kern_grads ():
            step_size ():
            b1 ():
            b2 ():
            eps ():

        Returns:

        """

        var, grad = var_and_grad
        if m is None:
            m = np.zeros(len(var))
        if v is None:
            v = np.zeros(len(var))

        m = self.b1* m + (1-self.b1)*grad
        v = self.b2*v + (1-self.b2)*np.square(v)
        alpha_t = self.step_size*np.sqrt(1-self.b2**t)/(1-self.b1**t)
        var = var + alpha_t*m/(np.sqrt(v) + self.eps)

        return var, m, v
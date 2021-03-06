import numpy as np


class CG:
    """
    Conjugate gradient "optimizer" used for solving linear systems
    """

    def __init__(self, cg_prod=None, tol=1e-3):
        """

        Args:
            cg_prod (): function for matrix vector products
            tol (): tolerance around solution
        """
        self.cg_prod = cg_prod
        self.tol = tol

    def cg(self, A, b, cg_prod=None, x=None, its=None):
        """
        runs CG procedure to solve Ax=b
        Args:
            A (): matrix
            b (): vector
            cg_prod (): mvp function
            x (): initial belief of solution
            its (): maximum iterations

        Returns:

        """

        n = len(b)

        if its is None:
            its = 2 * n
        if x is None:
            x = np.ones(n)
        if cg_prod is None:
            cg_prod = self.cg_prod

        r = cg_prod(A, x) - b
        p = - r
        r_k_norm = np.dot(r, r)
        for i in range(its):
            Ap = cg_prod(A, p)
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

class Adam:
    """
    Adam optimizer
    """
    def __init__(self, step_size=1e-3, b1=.9, b2=.99, eps=0.1):
        """

        Args:
            step_size (): initial step size
            b1 (): momentum
            b2 (): second order momentum
            eps (): jitter
        """
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self, var_and_grad, params):
        """
        Adapted from autograd.misc.optimizers
        Args:
            var_and_grad (): tuple (variable, gradient)
            params (): optimizer params to update

        Returns:

        """
        var, grad = var_and_grad
        if params is None:
            m = np.zeros(len(var))
            v = np.zeros(len(var))
            t = 1
        else:
            m, v, t = params
        m = self.b1 * m + (1 - self.b1) * grad
        v = self.b2 * v + (1 - self.b2) * np.square(v)
        alpha_t = self.step_size * \
                  np.sqrt(1 - self.b2 ** t)/(1 - self.b1 ** t)
        var = var + alpha_t * m/(np.sqrt(v) + self.eps)
        return var, (m, v, t + 1)


class SGD:
    """
    Stochastic gradient descent
    """
    def __init__(self, step_size=1e-3, momentum=0.9, decay=0.999):
        """

        Args:
            step_size ():
            momentum ():
            decay ():
        """
        self.step_size = step_size
        self.momentum = momentum
        self.decay=decay

    def step(self, var_and_grad, params):
        """
        Runs a step of SGD

        Args:
            var_and_grad (): tuple (variable, gradient)
            params (): optimizer params to update

        Returns:
        """
        if params is None:
            v_prev = 0.
            t = 1
        else:
            v_prev, t = params
        var, grad = var_and_grad
        v_t = (1 - self.momentum) * grad + self.momentum * v_prev
        return var + v_t * self.step_size * self.decay ** t, (v_t, t + 1)


class RMSProp:

    def __init__(self, step_size=1e-3, gamma=0.9, eps=1):
        self.step_size = step_size
        self.gamma = gamma
        self.eps = eps

    def step(self, var_and_grad, avg_sq_grad=None):

        var, grad = var_and_grad
        if avg_sq_grad is None:
            avg_sq_grad = np.ones(len(var))
        avg_sq_grad = avg_sq_grad * self.gamma + grad ** 2 * (1 - self.gamma)
        var = var + self.step_size * grad / (np.sqrt(avg_sq_grad) + self.eps)
        return var, avg_sq_grad
import autograd.numpy as np

"""
basic likelihood class
"""

class Poisson:
    """
    Implements Poisson likelihood with exponential link
    """
    def log_like(self, log_rate, y):

        return np.multiply(y, log_rate) - np.exp(log_rate)

class Bernoulli:

    """
    Implements Bernoulli likelihood with logistic sigmoid link
    """
    def log_like(self, g, y):

        return -np.multiply(y, np.log(1 + np.exp(-g))) - \
                              np.multiply(1 - y, np.log(1 + np.exp(g)))


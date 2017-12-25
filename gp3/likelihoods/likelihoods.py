import tensorflow as tf
import numpy as np

"""
basic likelihood class
"""

class PoissonLike:
    """
    Implements Poisson likelihood
    """
    def log_like(self, log_rate, y):

        return np.multiply(y, log_rate) - np.exp(log_rate)

    def grad(self, log_rate, y):

        return y - np.exp(log_rate)

    def hess(self, log_rate, y):

        return -np.exp(log_rate)

class BernoulliSigmoidLike:

    """
    Implements Bernoulli sigmoid likelihood
    """
    def log_like(self, y, g):

        return -tf.reduce_sum(tf.multiply(y, tf.log(1 + tf.exp(-g))) +
                              tf.multiply(1 - y, tf.log(1 + tf.exp(g))))


import tensorflow as tf
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
    def log_like(self, y, g):

        return -tf.reduce_sum(tf.multiply(y, tf.log(1 + tf.exp(-g))) +
                              tf.multiply(1 - y, tf.log(1 + tf.exp(g))))


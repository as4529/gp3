import autograd.numpy as np

def softplus(x):

    return np.log(np.exp(x) + 1)

def inv_softplus(x):

    return np.log(np.exp(x) - 1)
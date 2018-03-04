import autograd.numpy as np

def softplus(x):

    return np.log(np.exp(x) + 1)

def inv_softplus(x):

    return np.log(np.exp(x) - 1)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def unit_norm(x):
    x_sq = np.square(x)
    return x / np.sqrt(x_sq.sum(axis=1))[:,None]
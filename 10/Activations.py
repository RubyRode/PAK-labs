import numpy as np


def softmax(x):
    """Softmax function"""
    return np.exp(x) / np.sum(np.exp(x))


def relu(x):
    """Relu function"""
    return np.maximum(0, x)


def tanh(x):
    """Hyperbolic tangent function"""
    return np.tanh(x)

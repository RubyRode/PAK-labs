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


def sigmoid(x):
    """Sigmoid function"""
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    """"Sigmoid derivative function"""
    return x * (1 - x)
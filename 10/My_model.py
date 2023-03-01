import numpy as np


def softmax(x):
    """Softmax function for my model"""
    return np.exp(x) / np.sum(np.exp(x))


def relu(x):
    """Relu function for my model"""
    return np.maximum(0, x)


def tanh(x):
    """Hyperbolic tangent function for my model"""
    return np.tanh(x)


class Model:

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        """Model initialization"""
        self.weights_1 = np.random.randn(input_size)
        self.bias_1 = np.random.randn(hidden_size_1)
        self.weights_2 = np.random.randn(hidden_size_1)
        self.bias_2 = np.random.randn(hidden_size_2)
        self.weights_3 = np.random.randn(hidden_size_2)
        self.bias_3 = np.random.randn(output_size)
        self.act = {'soft': softmax, 'tanh': tanh, 'relu': relu}

    def forward(self, input_data, activation='relu'):
        """Forward propagation"""
        # np.squeeze(input_data)
        layer_1 = self.act[activation](np.dot(self.weights_1, input_data) + self.bias_1)
        layer_2 = self.act[activation](np.dot(self.weights_2, layer_1) + self.bias_2)
        layer_3 = self.act[activation](np.dot(self.weights_3, layer_2) + self.bias_3)
        return layer_3


if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.randn(256)
    in_outs = [256, 64, 16, 4]
    model = Model(*in_outs)
    res = model.forward(data, 'soft')
    print(res, '\n', sum(res))

import numpy as np
from Activations import relu, tanh, softmax


class Neuron:
    def __init__(self, size):
        self.input_size = size[0]
        self.output_size = size[1]
        self.weights = np.random.randn(self.input_size)
        self.bias = np.random.randn(self.output_size)
        self.act = {'soft': softmax, 'tanh': tanh, 'relu': relu}

    def forward(self, input_data, activator):
        return self.act[activator](np.dot(self.weights, input_data) + self.bias)


class Model:

    def __init__(self, sizes):
        """Model initialization"""
        self.sizes = sizes
        self.neurons = [Neuron(self.sizes[0])]
        for i in range(1, len(self.sizes)):
            self.neurons.append(Neuron(self.sizes[i]))

    def forward(self, input_data, activation='relu'):
        """Forward propagation"""
        result = input_data
        for neuron in self.neurons:
            result = neuron.forward(result, activation)
        return result


if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.randn(256)
    in_outs = [(256, 64), (64, 16), (16, 4)]
    model = Model(in_outs)
    res = model.forward(data, 'soft')
    print(res, '\n', sum(res))

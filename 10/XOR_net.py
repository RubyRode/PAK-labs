import numpy as np
from Activations import relu, tanh, softmax, sigmoid, sigmoid_derivative


class Neuron:

    def __init__(self, ins, outs):
        self._weights = np.random.uniform(size=(ins, outs))
        self._bias = np.random.uniform(1, outs)
        self.act = {'relu': relu, 'tanh': tanh, 'sigmoid': sigmoid, 'softmax': softmax}
        self.out = []

    def forward(self, x, activator='sigmoid'):
        layer_activation = np.dot(x, self._weights) + self._bias
        self.out = self.act[activator](layer_activation)
        return self.out

    def backward(self, x, loss):
        self._weights += np.dot(x, loss)
        self._bias += np.sum(loss, axis=0, keepdims=True)

    @property
    def weights(self):
        return self._weights


def accuracy(y, predict):
    return 1 - (np.abs(y - predict))


class Model:
    def __init__(self, input_layers: int, hidden_layers: int, output_layers: int):
        self.hidden_layers = Neuron(input_layers, hidden_layers)
        self.output_layers = Neuron(hidden_layers, output_layers)
        self.output = 0

    def forward(self, input):
        hidden_output = self.hidden_layers.forward(input)
        self.output = self.output_layers.forward(hidden_output)

        return self.output

    def backward(self, input, expected_output, predicted_output):
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layers = d_predicted_output.dot(self.output_layers.weights.T)
        d_hidden_layer = error_hidden_layers * sigmoid_derivative(self.hidden_layers.out)

        self.output_layers.backward(self.hidden_layers.out.T, d_predicted_output)
        self.hidden_layers.backward(input.T, d_hidden_layer)

    def train(self, X, y, epochs):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1

model = Model(input_size, hidden_size, output_size)
model.train(x, y, 10000)
output = model.predict([[0, 0], [1, 0]])

print(output, accuracy(([0], [1]), output))

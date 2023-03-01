import numpy as np
from numpy import ndarray


class Model:

    def __init__(self, X: ndarray):
        """Model initialization"""
        self.X = X
        self.outs = [64, 16, 4]
        self.shape = self.X.shape
        self.kernel = None
        self.act = {'relu': self.relu, 'softmax': self.softmax, 'tanh': self.tanh}

    def forward(self, activation='relu'):
        """Parameters convolution"""

        for j in self.outs:
            self.kernel = np.random.rand(self.X.shape[0] - j + 1)
            while self.X.shape[0] > j:
                self.X = np.convolve(self.X, self.kernel, 'valid')
                self.shape = self.X.shape
            self.act[activation]()
            print(self.X, "\n", self.X.shape, '\n')

    def relu(self):
        """Relu function for my model"""
        for j in range(0, self.X.shape[0]):
            self.X[j] = max(0, self.X[j])

    def softmax(self):
        """Softmax function for my model"""
        e_x = np.exp(self.X - np.max(self.X))
        self.X = e_x / e_x.sum()

    def tanh(self):
        """Hyperbolic tangent function for my model"""
        self.X = np.tanh(self.X)


if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.randn(256)
    print(data)
    some_model = Model(data)
    some_model.forward('softmax')
    print(sum(some_model.X))

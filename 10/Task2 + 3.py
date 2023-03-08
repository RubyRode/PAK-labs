import numpy as np
from Activations import relu, tanh, softmax
from Con_model import ConModel


class Layer_c:
    """Convolutional layer class"""
    def __init__(self, fil, shape, stride):
        """Convolutional layer construction"""
        self.fil = fil
        self.shape = shape
        self.stride = stride
        self.act = {'relu': relu, 'tanh': tanh, 'soft': softmax}

    def conv(self, matrix, activator='relu'):
        """Convolutional layer convolution"""
        filter_x, filter_y, filter_z, n_x  = self.fil.shape
        matrix_x, matrix_y, matrix_z = matrix.shape

        new_x = (matrix_x - filter_x) // self.stride + 1
        new_y = (matrix_y - filter_y) // self.stride + 1
        new_z = (matrix_z - filter_z) // self.stride + 1

        new_matrix = np.zeros(new_x * new_y * new_z * n_x).reshape((new_x, new_y, n_x))

        for n in range(0, n_x):
            for z in range(0, new_z, self.stride):
                for y in range(0, new_y, self.stride):
                    for x in range(0, new_x, self.stride):
                        new_matrix[y][x][n] = np.sum(matrix[y:y + filter_y, x:x + filter_x, z:z + filter_z]
                                                     * self.fil[:, :, :, n])

        return self.act[activator](new_matrix)

    def max_pool(self, matrix):
        """Max pooling function for a convoluted layer"""
        filter_x, filter_y, n_x = self.shape
        matrix_x, matrix_y, matrix_z = matrix.shape

        new_x = matrix_x // filter_x
        new_y = matrix_y // filter_y

        new_matrix = np.zeros(new_x * new_y * matrix_z).reshape((new_x, new_y, matrix_z))

        for n in range(0, n_x):
            for y in range(0, new_y, self.stride):
                for x in range(0, new_x, self.stride):
                    new_matrix[y][x][n] = np.max(matrix[y:y + filter_y, x:x + filter_x, n])

        return new_matrix


class Conv_model:
    """Convolutional model"""
    def __init__(self, layers_package):
        """All layers construction"""
        self.layers = []
        for (fil, shape, stride) in layers_package:
            self.layers.append(Layer_c(fil, shape, stride))

    def forward(self, matrix, activator='relu'):
        """Forward propagation through all layers"""
        for layer in self.layers:
            matrix = layer.conv(matrix, activator)
            matrix = layer.max_pool(matrix)
        return matrix


class Model:
    """Fully connected + Convolutional network class"""
    def __init__(self, layers_package, in_outs):
        """Fully connected + Convolutional networks initialization"""
        self.convLayer = Conv_model(layers_package)
        self.conLayer = ConModel(in_outs)

    def forward(self, in_, activator='relu'):
        """Forward propagation through all layers"""
        out = self.convLayer.forward(in_, activator)
        out = self.conLayer.forward(out.reshape(np.prod(out.shape), 1), activator)
        return out


if __name__ == "__main__":
    matrix = np.array(np.random.normal(scale=0.1, size=(19, 19, 3)))  # image for convolutional network

    # Filters for convolutional network
    fil1 = np.array(np.random.normal(size=(2, 2, 3, 8)))
    fil2 = np.array(np.random.normal(size=(2, 2, 8, 16)))
    # Pool size
    pool = (2, 2, 8)
    # Arguments package
    args = [(fil1, pool, 1), (fil2, pool, 1)]
    # Inner and outer sizes for fully connected network neurons
    in_outs = [(256, 64), (64, 16), (16, 4)]

    model = Model(args, in_outs)
    res = model.forward(matrix, 'soft')
    print(res, "\n", np.sum(res))

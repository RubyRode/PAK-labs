
import numpy as np
import matplotlib.pyplot as plt

def max_pool(input_data, pool_size):
    output_data = np.zeros(
        (input_data.shape[0] // pool_size, input_data.shape[1] // pool_size, input_data.shape[2]))
    for i in range(output_data.shape[0]):
        for j in range(output_data.shape[1]):
            for k in range(output_data.shape[2]):
                output_data[i, j, k] = np.max(
                    input_data[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size, k])
    return output_data


class ConvolutionalNetwork:
    def __init__(self, filter_size_1, num_filters_1, filter_size_2, num_filters_2):
        self.weights_1 = np.random.randn(filter_size_1, filter_size_1, filter_size_1, num_filters_1)
        self.bias_1 = np.random.randn(num_filters_1)

        self.weights_2 = np.random.randn(filter_size_2, filter_size_2, num_filters_1, num_filters_2)
        self.bias_2 = np.random.randn(num_filters_2)

    def forward_pass(self, input_data):
        pass


if __name__ == '__main__':
    cnn = ConvolutionalNetwork(filter_size_1=18, num_filters_1=8, filter_size_2=8, num_filters_2=16)
    input_data = np.random.randn(19, 19, 3)
    output_data = cnn.forward_pass(input_data)

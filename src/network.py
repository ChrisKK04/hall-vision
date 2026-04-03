"""
network.py

Implementation of an MLP neural network.
"""

import numpy as np

class Network:

    def __init__(self, sizes):
        self.sizes = sizes
        self.length = len(sizes)
        # the parameters are randomized using the normal distribution
        self.weights = [(np.random.randn(sizes[layer + 1], sizes[layer]))
                        for layer in range(len(sizes[1:]))]
        self.biases = [np.random.randn(layer, 1) for layer in sizes[1:]]

    def feedforward(self, a):
        # the zip is used to combine the weights and biases per layer into a list
        for weights, biases in zip(self.weights, self.biases):
            a = sigmoid(np.matmul(weights, a) + biases)
        return a
            
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    # random input vector of uniform distribution over [0, 1), shape (784, 1) - for testing
    a = np.random.rand(784, 1)

    net = Network([784, 64, 10])
    print(net.feedforward(a))
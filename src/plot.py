"""
plot.py

Plotting utilities
"""

import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

import data_loader
import network

training_data, validation_data, test_data = data_loader.load_mnist()

class PlotNetwork(network.Network):

    def __init__(self, sizes):
        self.sizes = sizes
        self.length = len(sizes)
        self.weights = [(np.random.randn(sizes[layer + 1], sizes[layer]))
                        for layer in range(len(sizes[1:]))]
        self.biases = [np.random.randn(layer, 1) for layer in sizes[1:]]
        self.accuracy_initial = 0
        self.accuracies = []

    def stochastic_gradient_descent(self, training_data, epochs, batch_size, training_rate, test_data):
        self.accuracy_initial = self.accuracy(test_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[end_index - batch_size:end_index]
                            for end_index in range(batch_size, len(training_data) + 1, batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, training_rate)
            self.accuracies.append(self.accuracy(test_data))
    
    def accuracy(self, test_data):
        correct_count = [1 for sample in test_data if np.argmax(self.feedforward(sample[0])) == sample[1]]
        return len(correct_count)

def extract_plot(network_structure, epochs, mini_batch_size, learning_rate):
    net = PlotNetwork(network_structure)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data)
    return (net.accuracy_initial, list(range(1, epochs + 1)), net.accuracies)

def write_plot(size, repeats, epochs, mini_batch_size, learning_rate):
    initial = []
    x = []
    y = []
    for i in range(repeats):
        print(f"Repeat {i + 1}/{repeats} for network structure {size}")
        initial_new, x_new, y_new = extract_plot(size, epochs, mini_batch_size, learning_rate)
        initial.append(initial_new)
        x = x + x_new
        y = y + y_new
    if True:
        with open(f"data/{str(size)}.pkl", "wb") as f:
            pickle.dump((initial, x, y), f)

def load_plot(structure):
    with open(f"data/{str(structure)}.pkl", "rb") as f:
        initial, x, y = pickle.load(f)

    plt.xticks(range(min(x), max(x) + 1))

    x = [x[i:i + 10] for i in range(0, len(x), 10)]
    y = [y[i:i + 10] for i in range(0, len(y), 10)]

    for i in range(len(x)):
        plt.plot(x[i], y[i])

    plt.title(f"{str(structure)} network (10 runs)")
    plt.xlim((0, 10 + 1))
    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    write_plot([784, 30, 10], repeats=10, epochs=10, mini_batch_size=20, learning_rate=5)
    load_plot([784, 30, 10])

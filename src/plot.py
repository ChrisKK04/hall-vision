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
        print(self.accuracy_initial)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[end_index - batch_size:end_index]
                            for end_index in range(batch_size, len(training_data) + 1, batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, training_rate)
            self.accuracies.append(self.accuracy(test_data))
            print(self.accuracies)
    
    def accuracy(self, test_data):
        correct_count = [1 for sample in test_data if np.argmax(self.feedforward(sample[0])) == sample[1]]
        return len(correct_count)

def extract_scatter(network_structure, epochs, mini_batch_size, learning_rate):
    net = PlotNetwork(network_structure)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data)
    return (net.accuracy_initial, list(range(1, epochs + 1)), net.accuracies)

def print_scatter(sizes, repeats, epochs, mini_batch_size, learning_rate):
    for size in sizes:
        initial = []
        x = []
        y = []
        for i in range(repeats):
            print(f"Repeat {i + 1} for {size}")
            initial_new, x_new, y_new = extract_scatter(size, epochs, mini_batch_size, learning_rate)
            initial.append(initial_new)
            x = x + x_new
            y = y + y_new
            print(initial, x, y)
        if True:
            with open(f"data/{str(size)}.pkl", "wb") as f:
                pickle.dump((initial, x, y), f)
        plt.scatter(x, y, label=str(size))
        
    plt.title("Hyperparameter comparison")
    plt.xlim((0, epochs + 1))
    plt.xticks(range(min(x), max(x) + 1))
    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.legend(title="Network structure")
    plt.show()

if __name__ == "__main__":
    print_scatter([[784, 10]], repeats=20, epochs=10, mini_batch_size=20, learning_rate=5)
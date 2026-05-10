"""
plot.py

Plotting utilities
"""

import math
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
    """Extracts accuracy information during neural network training.
    
    Args:
        network_structure (list): Layer structure of the network.
        epochs (int): Epoch count.
        mini_batch_size (int): Mini batch size.
        learning_rate (float): Learning rate.

    Returns:
        Tuple: (inital accuracy, list of [1, .., epochs], accuracy per epoch)
    """
    net = PlotNetwork(network_structure)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data)
    return (net.accuracy_initial, list(range(1, epochs + 1)), net.accuracies)

def write_plot(network_structure, repeats, epochs, mini_batch_size, learning_rate):
    """Writes network accuracy data into a file using multiple repeats.

    Args:
        network_structure (list): Layer structure of the network.
        repeats (int): Amount of times full training is repeated.
        epochs (int): Epoch count.
        mini_batch_size (int): Mini batch size.
        learning_rate (float): Learning rate.
    """
    initial = []
    x = []
    y = []
    for i in range(repeats):
        print(f"Repeat {i + 1}/{repeats} for network structure {network_structure}")
        initial_new, x_new, y_new = extract_plot(network_structure, epochs, mini_batch_size, learning_rate)
        initial.append(initial_new)
        x = x + x_new
        y = y + y_new
    if True:
        with open(f"data/{str(network_structure)}.pkl", "wb") as f:
            pickle.dump((initial, x, y), f)

def load_plot(network_structure):
    """Loads a plot from a file and prints a figure.

    Loads data from data/network_structure.pkl.
    
    Args:
        network_structure (list): Layer structure of the network.
    """
    with open(f"data/{str(network_structure)}.pkl", "rb") as f:
        initial, x, y = pickle.load(f)

    plt.xticks(range(min(x), max(x) + 1))

    x = [x[i:i + 10] for i in range(0, len(x), 10)]
    y = [y[i:i + 10] for i in range(0, len(y), 10)]

    for i in range(len(x)):
        plt.plot(x[i], y[i])

    plt.title(f"{str(network_structure)} network (10 runs)")
    plt.xlim((0, 10 + 1))
    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def convert_vectors_to_images(vectors):
    """Converts a list of 784-vectors into a list of 28x28-matrices (images)

    Args:
        vectors (list): List of 784-vectors

    Returns:
        List: each element is a tuple (image, digit)
            image (numpy.ndarray): 28x28 image
            digit (numpy.int64): digit
    """
    shape = (28, 28)
    images = []
    for vector in vectors:
        if type(vector[1]) is np.int64:
            images.append((vector[0].reshape(shape), vector[1]))
        else:
            images.append((vector[0].reshape(shape), np.argmax(vector[1])))
    return images

def show_images(images, classifications=False):
    """Prints figures of just MNIST images or alongside their neural classifications.
    
    Args:
        images (list): each image should be first converted with
            convert_vectors_to_images
        classifications (bool): Whether to include neural
            classifications for each of the printed images
    """
    # images = tuples of (28x28 matrix, digit)
    columns = 5
    rows = 5
    sets = math.ceil(len(images) / 25)
    for i in range(sets):
        fig = plt.figure(figsize=(7, 7))
        fig.suptitle(f"Set {i+1}/{sets}")
        # include subtext
        if classifications:
            fig.text(0.5, 0.05, "(digit, network)", ha="center")
        for j in range(1, min(25, len(images)) + 1):
            fig.add_subplot(rows, columns, j)
            # include network classification
            if classifications:
                plt.title(f"{int(images[j-1][1])}, {int(images[j-1][2])}")
            else:
                plt.title(int(images[j-1][1]))
            plt.imshow(images[j-1][0], cmap="gray")
            plt.axis("off")
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        images = images[25:]

if __name__ == "__main__":
    write_plot([784, 30, 10], repeats=10, epochs=10, mini_batch_size=20, learning_rate=5)
    load_plot([784, 30, 10])

    show_images(convert_vectors_to_images(training_data[:70]))

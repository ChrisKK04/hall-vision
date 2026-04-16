"""
data_loader.py

Functions for loading data

load_mnist() - loads the MNIST dataset as tuple
(training_data, validation_data, test_data).

load_parameters() - loads the parameters for use
in inference.
"""

import math
import random

import gzip
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_mnist():
    with gzip.open("data/mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f)
    return (training_data, validation_data, test_data)

def load_mnist_test():
    training_data, validation_data, test_data = load_mnist()
    return test_data

def load_network_parameters():
    with open("data/parameters.pkl", "rb") as f:
        sizes, weights, biases = pickle.load(f)
    return (sizes, weights, biases)

def convert_vectors_to_images(vectors):
    """Converts 784-vectors into 28x28-matrices (images)"""
    shape = (28, 28)
    images = []
    for vector in vectors:
        if type(vector[1]) is np.int64:
            images.append((vector[0].reshape(shape), vector[1]))
        else:
            images.append((vector[0].reshape(shape), np.argmax(vector[1])))
    return images

def show_images(images):
    """Prints figures of MNIST images"""
    # images = tuples of (28x28 matrix, digit)
    columns = 5
    rows = 5
    sets = math.ceil(len(images) / 25)
    for i in range(sets):
        fig = plt.figure(figsize=(7, 7))
        for j in range(1, min(25, len(images)) + 1):
            fig.suptitle(f"Set {i+1}/{sets}")
            fig.add_subplot(rows, columns, j)
            plt.imshow(images[j-1][0], cmap="gray")
            plt.title(int(images[j-1][1]))
            plt.axis("off")
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        images = images[25:]

if __name__ == "__main__":
    training_data, validation_data, test_data = load_mnist()
    random.shuffle(training_data)
    show_images(convert_vectors_to_images(training_data[:70]))
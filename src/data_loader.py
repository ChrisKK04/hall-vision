"""
data_loader.py

Functions for loading data

load_mnist() - loads the MNIST dataset as tuple
(training_data, validation_data, test_data).

load_parameters() - loads the parameters for use
in inference.
"""

import gzip
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
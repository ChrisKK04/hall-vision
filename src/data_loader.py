"""
data_loader.py

Functions for loading data

load_mnist() - loads the MNIST dataset as tuple
(training_data, validation_data, test_data).

load_mnist_test() - loads the MNIST test dataset
as a list of images and digits.

load_parameters() - loads the parameters for use
in inference.
"""

import gzip
import pickle

def load_mnist():
    """Loads MNIST data.

    Returns:
        Tuple: (training_data, validation_data, test_data),
            where each element is a list of image-label pairs.
    """
    with gzip.open("data/mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f)
    return (training_data, validation_data, test_data)

def load_mnist_test():
    """Loads MNIST test data

    Returns:
        List: test_data with image-label pairs.
    """
    training_data, validation_data, test_data = load_mnist()
    return test_data

def load_network_parameters(path="data/parameters.pkl"):
    """Loads neural network parameters from a given file.

    Args:
        path (str): Path to the network parameter file.

    Returns:
        tuple: (sizes, weights, biases)
            - sizes (list of int): Dimensions of each layer.
            - weights (list of numpy.ndarray): Weight matrices for each layer.
            - biases (list of numpy.ndarray): Bias vectors for each layer.
    """
    with open(str(path), "rb") as f:
        sizes, weights, biases = pickle.load(f)
    return (sizes, weights, biases)

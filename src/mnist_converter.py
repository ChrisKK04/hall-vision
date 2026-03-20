"""
mnist_converter.py

This script reshapes the image arrays and integers in
data/mnist_original.pkl.gz into arrays for easier training.

The file used in training is the output file mnist_pkl.gz.
"""

import pickle
import gzip

import numpy as np

def load_data():
    with gzip.open("data/mnist_original.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    with gzip.open("data/mnist.pkl.gz", "wb") as f:
        pickle.dump((training_data, validation_data, test_data), f)
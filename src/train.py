"""
train.py

A python script used for the training neural
network for use in MNIST digit classification.

Can be used to extract the parameters
(sizes, biases, weights).

sizes = layer structure of the network.
"""

import pickle

import data_loader
import network

def train(network_structure: list, epochs: int, mini_batch_size: int,
          learning_rate: float, extract_parameters=True, testing=True):
    # load the training, validation and test data
    training_data, validation_data, test_data = data_loader.load_mnist()

    # network structure [input layer, hidden layer, ..., hidden layer, output layer]
    net = network.Network(network_structure)

    # train with: training data, epochs, mini batch size, learning rate, testing data
    if testing:
        net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data)
    else:
        net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate)

    # extract biases and weights into a file
    extract_parameters = extract_parameters
    if extract_parameters:
        sizes, weights, biases = net.extract_parameters()
        with open("data/parameters.pkl", "wb") as f:
            pickle.dump((sizes, weights, biases), f)

if __name__ == "__main__":
    train(network_structure=[784, 30, 10], epochs=10, mini_batch_size=30,
          learning_rate=5, extract_parameters=False, testing=True)
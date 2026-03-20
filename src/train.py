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

def train():
    # load the training, validation and test data
    training_data, validation_data, test_data = data_loader.load_mnist()

    # network structure [input layer, hidden layer, ..., hidden layer, output layer]
    net = network.Network([784, 10])

    # train with: training data, epochs, mini batch size, learning rate, testing data
    net.SGD(training_data, 1, 10, 3.0, test_data=test_data)

    # extract biases and weights into a file
    extract_parameters = True
    if extract_parameters:
        sizes, biases, weights = net.extract_parameters()
        with open("data/parameters.pkl", "wb") as f:
            pickle.dump((sizes, biases, weights), f)

if __name__ == "__main__":
    train()
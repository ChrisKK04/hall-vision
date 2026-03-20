"""
inference.py

A python script for running inference on
a trained neural network.
"""

import data_loader
import network

def inference():
    # load the network parameters
    sizes, biases, weights = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_biases_weights(biases, weights)
    test_data = data_loader.load_mnist_test()

    print(f"Correct classifications: {net.evaluate(test_data)} / {len(test_data)}")

if __name__ == "__main__":
    inference()
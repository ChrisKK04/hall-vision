"""
inference.py

A python script for running inference on
a trained neural network.
"""

import data_loader
import network
import preprocessing

def inference():
    # load the network parameters
    sizes, biases, weights = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_biases_weights(biases, weights)
    test_data = data_loader.load_mnist_test()

    print(f"Correct classifications: {net.evaluate(test_data)} / {len(test_data)}")

def inference_custom_image(img_path, digit, threshold=255, show=False):
    sizes, biases, weights = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_biases_weights(biases, weights)
    input = preprocessing.preprocess(img_path, digit, threshold, show)

    print(f"Neural classification: {net.evaluate_single(input)} Actual classification: {digit}")

if __name__ == "__main__":
    inference_custom_image("images/five.png", 5, 200, True)
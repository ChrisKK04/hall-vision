"""
inference.py

A python script for running inference on
a trained neural network.
"""

import data_loader
import network
import preprocessing

def inference():
    sizes, weights, biases = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_weights_biases(weights, biases)
    test_data = data_loader.load_mnist_test()
    accuracy = net.accuracy(test_data)

    print(f"Correct classifications: {accuracy[0]} / {accuracy[1]}")

def inference_custom_image(img_path, digit, threshold=255, show=False):
    sizes, weights, biases = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_weights_biases(weights, biases)
    input = preprocessing.preprocess(img_path, digit, threshold, show)

    print(f"Neural classification: {net.evaluate_single(input[0])}, Actual classification: {digit}")

if __name__ == "__main__":
    inference_custom_image(img_path="images/zero.png", digit=0, threshold=200, show=False)
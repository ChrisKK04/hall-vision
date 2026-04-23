"""
inference.py

A python script for running inference on a trained neural network.
Includes utilities for testing on the MNIST test dataset and single image classification.
"""

import data_loader
import network
import preprocessing

def inference_on_test(figure=False):
    """Runs inference on the MNIST test dataset"""
    sizes, weights, biases = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_weights_biases(weights, biases)
    test_data = data_loader.load_mnist_test()
    accuracy = net.accuracy(test_data)

    print(f"Correct classifications: {accuracy[0]} / {accuracy[1]}")

    # prints a figure that includes the incorrect classifications
    if figure:
        incorrect = net.extract_incorrect(test_data)
        converted = data_loader.convert_vectors_to_images(incorrect)
        images = []
        for i in range(len(incorrect)):
            images.append((converted[i][0], converted[i][1], incorrect[i][2]))
        data_loader.show_images(images, classifications=True)

def inference_custom_image(img_path, digit, threshold=255, show=False):
    """Runs inference on a single input image"""
    sizes, weights, biases = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_weights_biases(weights, biases)
    input = preprocessing.preprocess(img_path, digit, threshold, show)

    print(f"Neural classification: {net.evaluate_single(input[0])}, Actual classification: {digit}")

if __name__ == "__main__":
    inference_on_test(figure=True)
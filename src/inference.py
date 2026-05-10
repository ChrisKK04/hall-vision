"""
inference.py

A python script for running inference on a trained neural network.

inference_on_test() - runs inference on the MNIST
test dataset.

infrence_custom_image() - runs inference on a
provided image.
"""

import data_loader
import network
import plot
import preprocessing

def inference_on_test(figure=False):
    """Runs inference on the MNIST test dataset.

    Prints network accuracy and figure (if selected).
    
    Args:
        figure (bool): Whether to create a figure of
            incorrect classifications.
    """
    sizes, weights, biases = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_weights_biases(weights, biases)
    test_data = data_loader.load_mnist_test()
    accuracy = net.accuracy(test_data)

    print(f"Correct classifications: {accuracy[0]} / {accuracy[1]}")

    # prints a figure that includes the incorrect classifications
    if figure:
        incorrect = net.extract_incorrect(test_data)
        converted = plot.convert_vectors_to_images(incorrect)
        images = []
        for i in range(len(incorrect)):
            images.append((converted[i][0], converted[i][1], incorrect[i][2]))
        plot.show_images(images, classifications=True)

def inference_custom_image(img_path, digit, threshold=255, show=False):
    """Runs inference on a single input image

    Prints neural classification of the given image
    
    Args:
        img_path (str): Path to the image.
        digit (int): The real classification of the digit.
        threshold (int): 0-255, lower values remove background imperfections.
        show (bool): Whether to print the converted image.
    """
    sizes, weights, biases = data_loader.load_network_parameters()

    net = network.Network(sizes)
    net.set_weights_biases(weights, biases)
    input = preprocessing.preprocess(img_path, digit, threshold, show)

    print(f"Neural classification: {net.evaluate_single(input[0])}, Actual classification: {digit}")

if __name__ == "__main__":
    inference_on_test(figure=True)
    inference_custom_image(img_path="test_images/image.png", digit=2, threshold=255, show=True)

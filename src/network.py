"""
network.py

Implementation of an MLP neural network using
stochastic gradient descent and the sigmoid
activation function.

BaseNetwork - the core architecture of the neural network
Network - extra functionality for use when using the network
"""

import random

import numpy as np

# the core of the neural network architecture
class BaseNetwork:

    def __init__(self, sizes: list):
        """Initializes a network using a list of the given structure"""
        self.sizes = sizes
        self.length = len(sizes)
        self.weights = [(np.random.randn(sizes[layer + 1], sizes[layer]))
                        for layer in range(len(sizes[1:]))]
        self.biases = [np.random.randn(layer, 1) for layer in sizes[1:]]
    
    def feedforward(self, a):
        for weights, biases in zip(self.weights, self.biases):
            a = self.sigmoid(np.matmul(weights, a) + biases)
        return a

    def feedforward_save(self, a):
        # activations and weighted inputs into lists per layer
        activations = [a]
        weighted_inputs = []
        for weights, biases in zip(self.weights, self.biases):
            z = np.matmul(weights, a) + biases
            weighted_inputs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        return activations, weighted_inputs
    
    def stochastic_gradient_descent(self, training_data: list, epochs: int, batch_size: int, training_rate: int):
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[end_index - batch_size:end_index]
                            for end_index in range(batch_size, len(training_data) + 1, batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, training_rate)

    def update_parameters(self, mini_batch, training_rate):
        # activations = layer: 1, 2, ..., L - 1
        # errors = layer: 2, 3, ..., L
        activations = []
        errors = []
        for sample in mini_batch:
            activation, error = self.backpropagate(sample)
            activations.append(activation)
            errors.append(error)

        # update parameters layer by layer starting from first hidden layer
        for layer in range(self.length - 1):
            # null vectors for the sum
            error_activation_vector = np.zeros(np.shape(self.weights[layer]))
            error_vector = np.zeros(np.shape(self.biases[layer]))
            # sum the vectors for the layer
            for error, activation in zip(errors, activations):
                error_activation_vector = error_activation_vector + np.matmul(error[layer], np.transpose(activation[layer]))
                error_vector = error_vector + error[layer]
            # update the parameters
            self.weights[layer] = self.weights[layer] - (np.multiply((training_rate / len(mini_batch)), error_activation_vector))
            self.biases[layer] = self.biases[layer] - (np.multiply((training_rate / len(mini_batch)), error_vector))

    def backpropagate(self, sample):
        # sample[0] = input activation vector (image), sample[1] = desired output vector (digit)
        # feedforward the image input
        activations, weighted_inputs = self.feedforward_save(sample[0])
        # last layer error
        errors = [np.multiply(activations[-1] - sample[1], self.sigmoid_prime(weighted_inputs[-1]))]
        # other layer errors
        for layer in range(2, len(weighted_inputs) + 1):
            errors.append(np.multiply(np.matmul(np.transpose(self.weights[-layer + 1]), errors[-1]), self.sigmoid_prime(weighted_inputs[-layer])))
        errors.reverse()
        return activations, errors

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

# The main class used when using the network
# includes extra functionality not core to the learning
class Network(BaseNetwork):
    
    # reimplemented with accuracy testing after each epoch
    def stochastic_gradient_descent(self, training_data, epochs, batch_size, training_rate, test_data=False):
        """SGD with post epoch prints"""
        if test_data:
            accuracy = self.accuracy(test_data)
            print(f"Epoch {epoch}: {accuracy[0]}/{accuracy[1]}")
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[end_index - batch_size:end_index]
                            for end_index in range(batch_size, len(training_data) + 1, batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, training_rate)
            if test_data:
                accuracy = self.accuracy(test_data)
                print(f"Epoch {epoch + 1}: {accuracy[0]}/{accuracy[1]}")
    
    def accuracy(self, test_data):
        """Test accuracy on test data"""
        if type(test_data[0][1]) is np.int64:
            correct_count = [1 for sample in test_data if np.argmax(self.feedforward(sample[0])) == sample[1]]
        else:
            correct_count = [1 for sample in test_data if np.argmax(self.feedforward(sample[0])) == np.argmax(sample[1])]
        return (len(correct_count), len(test_data))
    
    def evaluate_single(self, image):
        """Evaluates a single image"""
        return np.argmax(self.feedforward(image))
    
    def set_weights_biases(self, weights, biases):
        """Sets weights and biases"""
        self.weights = weights
        self.biases = biases

    def extract_parameters(self) -> tuple[list, list, list]:
        """Returns tuple of (network shape, weights, biases)"""
        return (self.sizes.copy(), self.weights.copy(), self.biases.copy())
    
    def extract_incorrect(self, data):
        """Returns the images with the wrong classification
        [(vector, digit)] -> [(vector, digit, classification)]"""
        incorrect = []
        for sample in data:
            classification = np.argmax(self.feedforward(sample[0]))
            if classification != sample[1]:
                incorrect.append((sample[0], sample[1], classification))
        return incorrect
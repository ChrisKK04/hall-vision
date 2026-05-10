"""
network.py

Implementation of an MLP neural network using
stochastic gradient descent and the sigmoid
activation function.

BaseNetwork - the core architecture of the neural network.
Network - extra functionality for use when using the network.
"""

import random

import numpy as np

# the core of the neural network architecture
class BaseNetwork:
    """The core neural network architecture class used in the project"""

    def __init__(self, sizes):
        """Initializes a neural network using the given structure.
        
        Args:
            sizes (list): a list of integers used for defining
                the neural network's layer structure.
        """
        self.sizes = sizes
        self.length = len(sizes)
        self.weights = [(np.random.randn(sizes[layer + 1], sizes[layer]))
                        for layer in range(len(sizes[1:]))]
        self.biases = [np.random.randn(layer, 1) for layer in sizes[1:]]

    def feedforward(self, a):
        """Feedforwards input activations.

        Args:
            a (numpy.ndarray): input activations.

        Returns:
            numpy.ndarray: array of last layer activations.
        """
        for weights, biases in zip(self.weights, self.biases):
            a = self.sigmoid(np.matmul(weights, a) + biases)
        return a

    def feedforward_save(self, a):
        """Saves activations and weighted inputs into lists per layer.

        Args:
            a (numpy.ndarray): input activations.

        Returns:
            Activations (list): activations per layer.
            Weighted_inputs (list): weighted_inputs per layer.
        """
        activations = [a]
        weighted_inputs = []
        for weights, biases in zip(self.weights, self.biases):
            z = np.matmul(weights, a) + biases
            weighted_inputs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        return activations, weighted_inputs

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate):
        """Stochastic gradient descent algorithm.

        Uses SGD to update the weights and biases of the initialized neural network.
        
        Args:
            training_data (list): list of tuples (image, digit)
            epochs (int): Epoch count.
            mini_batch_size (int): Mini batch size.
            learning_rate (float): Learning rate.
        """
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[end_index - mini_batch_size:end_index]
                            for end_index in range(mini_batch_size, len(training_data) + 1, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, learning_rate)

    def update_parameters(self, mini_batch, learning_rate):
        """Updates network parameters using backpropagation
        
        Args:
            mini_batch (list): a batch of training samples
            learning_rate (float): Learning rate.
        """
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
            self.weights[layer] = self.weights[layer] - (np.multiply((learning_rate / len(mini_batch)), error_activation_vector))
            self.biases[layer] = self.biases[layer] - (np.multiply((learning_rate / len(mini_batch)), error_vector))

    def backpropagate(self, sample):
        """Backpropagation algorithm.

        Runs backpropgation on a single training sample.
        
        Args:
            sample (tuple): (image, digit)
        """
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
        """Sigmoid activation function.
        
        Args:
            z (numpy.ndarray): weighted input.

        Returns:
            numpy.ndarray: activation.
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function.

        Args:
            z (numpy.ndarray): weighted input.
        
        Returns:
            numpy.ndarray: derivative of the sigmoid function.
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

class Network(BaseNetwork):
    """The main neural network class used when using the network
    
    Includes extra functionality not core to the learning algorithm.
    """

    # reimplemented with accuracy testing after each epoch
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=False):
        """SGD with an option for post epoch prints.

        Args:
            training_data (list): list of tuples (image, digit)
            epochs (int): Epoch count.
            mini_batch_size (int): Mini batch size.
            learning_rate (float): Learning rate.
            test-data (bool): if true, network accuracy is
                printed after each epoch (also once before any epochs).
        """
        if test_data:
            accuracy = self.accuracy(test_data)
            print(f"Epoch 0: {accuracy[0]}/{accuracy[1]}")
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[end_index - mini_batch_size:end_index]
                            for end_index in range(mini_batch_size, len(training_data) + 1, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, learning_rate)
            if test_data:
                accuracy = self.accuracy(test_data)
                print(f"Epoch {epoch + 1}: {accuracy[0]}/{accuracy[1]}")

    def accuracy(self, test_data):
        """Tests network accuracy on test data.
        
        Args:
            test_data (list): (image, digit).
        
        Return:
            Tuple: (correct classification count, length of test data).
        """
        if type(test_data[0][1]) is np.int64:
            correct_count = [1 for sample in test_data if np.argmax(self.feedforward(sample[0])) == sample[1]]
        else:
            correct_count = [1 for sample in test_data if np.argmax(self.feedforward(sample[0])) == np.argmax(sample[1])]
        return (len(correct_count), len(test_data))

    def evaluate_single(self, image):
        """Evaluates a single image.
        
        Args:
            image (numpy.ndarray): image used for evalution.

        Returns:
            numpy.int64: neural classification of the image.
        """
        return np.argmax(self.feedforward(image))

    def set_weights_biases(self, weights, biases):
        """Sets weights and biases of network.
        
        Args:
            weights (list): weight matrices per layer.
            biases (list): bias vectors per layer.
        """
        self.weights = weights
        self.biases = biases

    def extract_parameters(self) -> tuple[list, list, list]:
        """Extracts the network's parameters

        Returns:
            Tuple: (sizes, weights, biases)
                sizes (list): network layer structure
                weights (list): weight matrices per layer. 
                biases (list): bias vectors per layer.
        """
        return (self.sizes.copy(), self.weights.copy(), self.biases.copy())

    def extract_incorrect(self, data):
        """Runs inference on a dataset and returns incorrect classifications.
        
        Args:
            data (list): (vector, digit).
                vector: image as a 784-vector.
                digit: correct classification.
        
        Returns:
            List: (vector, digit, neural classification)
                vectr: image as a 784-vector.
                digit: correct classification.
                neural classification: classifications selected by
                    the neural network.
        """
        incorrect = []
        for sample in data:
            classification = np.argmax(self.feedforward(sample[0]))
            if classification != sample[1]:
                incorrect.append((sample[0], sample[1], classification))
        return incorrect

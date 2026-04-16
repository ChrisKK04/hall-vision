"""
test_network.py

File for automated tests of the used neural network

Run the tests below with "python3 -m pytest tests/test_network.py"
from the root of the project.

"poetry run pytest --cov=src tests/" - coverage
"""

import random

import numpy as np
import pytest

from src import data_loader
from src import network

# set fixed seeds for each run
@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(1337) # sample shuffling
    np.random.seed(1337) # parameter initialization

# overfitting on a small dataset (learn everything)
def test_overfit():
    training_data, validation_data, test_data = data_loader.load_mnist()
    training_data = training_data[:10]

    net = network.Network([784, 10, 10])
    beginning = net.accuracy(training_data)
    net.stochastic_gradient_descent(training_data, 300, 10, 3)
    end = net.accuracy(training_data)
    counter = 0
    # network improves
    if beginning[0] != end[0]:
        counter += 1
    # overfit
    if end[0] == end[1]:
        counter += 1

    assert counter == 2

# network parameters update after each mini batch
def test_all_layers_update():
    class TestNetwork(network.Network):
        # Network implementation but passes parameters after each mini batch
        # only lines with comments above are changed from original implementation
        def stochastic_gradient_descent(self, training_data, epochs, batch_size, training_rate, test_data=False):
            # initial parameters
            sizes_old, weigths_old, biases_old = self.extract_parameters()
            # layers - 1 = only layers with parameters
            structure = len(sizes_old) - 1
            for epoch in range(epochs):
                random.shuffle(training_data)
                mini_batches = [training_data[end_index - batch_size:end_index]
                                for end_index in range(batch_size, len(training_data) + 1, batch_size)]
                for mini_batch in mini_batches:
                    self.update_parameters(mini_batch, training_rate)
                    sizes_new, weights_new, biases_new = self.extract_parameters()
                    # weights and biases change check after each update
                    for i in range(structure):
                        assert np.array_equal(weigths_old[i], weights_new[i]) == False
                        assert np.array_equal(biases_old[i], biases_new[i]) == False
                    # update previous
                    sizes_old, weigths_old, biases_old = sizes_new, weights_new, biases_new
                if test_data:
                    accuracy = self.accuracy(test_data)
                    print(f"Epoch {epoch + 1}: {accuracy[0]}/{accuracy[1]}")

    training_data, validation_data, test_data = data_loader.load_mnist()
    training_data = training_data[:10]
    net = TestNetwork([784, 256, 128, 64, 32, 16, 10])
    net.stochastic_gradient_descent(training_data, 10, 5, 3)

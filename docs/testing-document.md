# Testing

## Introduction
Testing even small neural networks is inherently hard due to them being based on weights and biases, for which no inherently correct values exist. Due to the weights and biases being calculated deterministically, the correct calculations for these based on the architecture could be checked but this would realistically only be possible for **very** small networks with tiny inputs where the "correct" weights and biases could be easily computed and checked. This isn't feasible for even the tiny network used in the project. Thus neural network testing is usually based on checking if the network is doing "something". The testing for this project was done by implementing all of the test examples listed in [this](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/) article about testing neural networks.

## The tests
The implemented tests can be found in the [tests](/tests) folder with the tests, testing the following:

- **Overfitting** (test_overfit): If a neural network overfits (can achieve a 100% accuracy on dataset), the network is learning something, thus the learning pipeline is working and parameters are being updated. The implemented test keeps training a neural network on 10 training images until it overfits. Uses a [784, 10, 10] network.

- **Utilization of all network layers** (test_all_layers_update): For the best accuracy, a neural network should use and update all of it's layers. The implemented test checks that the parameters of each network layer update (don't match the previous ones) after each mini batch of training. Uses a [784, 256, 128, 64, 32, 16, 10] network.

- **Correct network shape** (test_network_shape): The actual structure of a neural network should match the structure specified by the programmer. The implemented test checks that the specified network structure matches the actual structure. Uses a [784, 256, 128, 64, 32, 16, 10] network

## Coverage report

```
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
src\data_loader.py          14      5    64%   23-24, 27-29
src\inference.py            25     25     0%   8-43
src\mnist_converter.py      25     25     0%   16-45
src\network.py              85     19    78%   44-49, 100-101, 109-110, 115, 122, 126-127, 136-141
src\plot.py                102    102     0%   7-137
src\preprocessing.py        15     15     0%   1-27
src\train.py                19     19     0%   13-45
------------------------------------------------------
TOTAL                      285    210    26%
```

**Note: Testing of the project was only done to the neural network portion of the project due to it being the **main** focus of the project. The missing lines in network.py are not core to the implemented learning algorithm.**

## Running the tests
**(whilst being in the root of the project and not in the poetry virtual environment)**

The tests can be ran with:
```
poetry run python3 -m pytest tests/test_network.py
```

Line coverage can be ran with:
```
poetry run python3 -m pytest --cov=src tests/
```
The line coverage report can be obtained with:
```
poetry run coverage report -m
```
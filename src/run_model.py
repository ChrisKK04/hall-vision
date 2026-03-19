import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10]) # network structure
# data, epochs, mini batch size, learning rate, testing data
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
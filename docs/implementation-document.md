# Implementation

## Project structure
The program is seperated into multiple different folders and files, each with it's own purpose. Below is a list of the folders and their contents.

data:
- `mnist_original.pkl.gz`: The original unconverted version of the MNIST dataset.

- `mnist.pkl.gz`: The used (converted) MNIST dataset.

- `parameters.pkl`: The (default) location of extracted network parameters. By default, includes the parameters which achieved the best accuracy during development.

- The `[784, 30, 10]` file and other files like it, contain data on different network layer architecture accuracies. The data in the files can be plotted with the load_plot()-function in `plot.py`

docs:
- The `weekly-reports` folder includes weekly reports made during the course of the project.

- `implementation-document.md`: Info on the implementation of the project.

- `specification-document.md`: A specification document made at the beginning of the project.

- `testing-document.md`: Info on the automated tests of the project.

- `user-guide.md`: The user guide.

images:
- Images of the project (classification accuracy plots and MNIST digit and classification examples).

src:
- `data_loader.py`: Functions for loading the MNIST data and network parameters.

- `inference.py`: Inference functions for running inference on the test dataset and on a custom image.

- `mnist_converter.py`: A data converter used for converting the originally used MNIST dataset.

- `network.py`: The neural network of the project. Includes classes BaseNetwork (the core of the neural network) and Network (a version of the BaseNetwork class used actually used for inference and training).

- `plot.py`: Functions for plotting neural network performance and figures of image predictions.

- `preprocessing.py`: The preprocessing pipeline used when running inference on a custom image.

- `train.py`: Training function for training a neural network with the given hyperparamaters.

tests:
- `test_network.py`: Tests related to testing the neural network.

## Network accuracy

The best hyperparameters, through extensive experimentation, were determined to be a layer structure of [784, 30, 10], 30 epochs, a mini batch size of 20 and a learning rate of 5, which totals to 23 860 parameters. The best test dataset accuracy reached with these hyperparameters was 9623 / 10 000 (96,23%). The parameters can be found in [/data/parameters.pkl](/data/parameters.pkl).

## Shortcomings and improvements
The usability of the project could be improved by providing more complex usage of the program via the command line, as currently the program is configured through changing the source code. Of course, the neural network itself could be massively improved by adding more features to it (for example as is covered in chapter 3 [here](http://neuralnetworksanddeeplearning.com/chap3.html)) or by changing the architecture to a contemporary one for digit recognition. The matrix operations run on a CPU (NumPy library), and thus, massive speedups of especially training could be achieved through using a GPU.

## The use of large language models (LLMs)
Large language models, specifically [Google's Gemini](https://gemini.google.com/) and [OpenAI's ChatGPT](https://chatgpt.com/) (Online versions, not IDE versions), were used to figure out plotting using Matplotlib, for the HTML image formatting in the README and to explain confusing neural network topics. LLMs weren't used in the writing of any of the code or documentation.

## Sources
- (main source) Michael Nielsen's online book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

- (testing) Sebastian Björkqvist's article [Writing automated tests for neural networks](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)

- (libraries) [NumPy](https://numpy.org/) (fast matrix operations), [OpenCV](https://opencv.org/) (image preprocessing), [Matplotlib](https://matplotlib.org/) (printing images), [Pytest](https://docs.pytest.org/en/stable/) (automatic testing) and [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) (Pytest test coverage)
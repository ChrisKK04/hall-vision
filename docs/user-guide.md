# User guide

Note: The program is used by modifying the python files and running them. The Images section of the user guide includes guidance on detecting custom handwritten digits using the neural network. The dependencies are downloaded using Poetry.

### Basics
The main usage of the program is done through [train.py](/src/train.py) and [inference.py](/src/inference.py)

**[train.py](/src/train.py)**  
Used for training. The neural network can be trained using the train() function by defining the network structure, the amount of epochs, the mini batch size and the learning rate. Also, the parameters of the network can be extracted and testing can be turned on or off (shows classification accuracy each epoch). If parameter extracting is enabled by default, the parameters will be saved into /data/parameters.pkl.

```
train(network_structure=[784, 30, 10], epochs=10, mini_batch_size=30, learning_rate=5, extract_parameters=False, testing=True)
```

**[inference.py](/src/inference.py)**  
Used for inference. inference() will initialize a network, load extracted parameters into it and run inference on the MNIST test data, with the function printing the classification accuracy.

```
inference()
```

Inference can also be ran on custom user submitted images using the inference_custom_image() function in inference(). The function requires the image path, the actual digit and the threshold (used to remove imperfections in the background of the digit). The function will then run preprocessing on the image and then classify the image. The show-parameter in the function can be used to print the preprocessed image.

```
inference_custom_image(img_path="images/zero.png", digit=0, threshold=200, show=False)
```

### Advanced
**[network.py](/src/network.py)**  
The core of the used neural network. The sigmoid function is used in neuron activations and stochastic gradient descent is the used learning algorithm. The neural network implementation is based on the descriptions found in the online book [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), specifically chapters 1 and 2.

**[data_loader.py](/src/data_loader.py)**  
This is a loader module used for loading the MNIST dataset and network parameters.

**[mnist_converter.py](/src/mnist_converter.py)**  
The conversion that was done on the MNIST dataset. See [data-info.md](/data/data-info.md).

**[preprocessing.py](/src/preprocessing.py)**  
The preprocessing pipeline used to make a user submitted image ready for inference. The image is turned into grayscale, resized to 28x28, background noise is removed, pixel values are normalized to range 0 - 1, pixels are flipped (black to white and white to black) and finally the image is shaped into a (784, 1) NumPy matrix.

Information on the used MNIST dataset can be found in [data-info.md](/data/data-info.md).

### Images
The user submitted images should only include the digit itself. The digit should be drawn in black on a white background, which should include little noise. The digit should also be centered and upright oriented.
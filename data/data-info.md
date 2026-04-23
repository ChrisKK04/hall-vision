The file [mnist.pkl.gz](data/mnist.pkl.gz) includes the MNIST dataset in a form that is easy to load and manipulate using Python. The file was converted from [mnist_original.pkl.gz](/data/mnist_original.pkl.gz), which was originally obtained from chapter 1 in [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) from the companion [repository](https://github.com/mnielsen/neural-networks-and-deep-learning). The author of the book originally obtained the file dataset from the LISA machine learning laboratory at the University of Montreal ([link](http://www.deeplearning.net/tutorial/gettingstarted.html) - seemingly defunct).

The dataset uses [pickling](https://docs.python.org/3/library/pickle.html) (.pkl) to allow the loading of the images directly as NumPy arrays. The dataset was further compressed using [gzip](https://docs.python.org/3/library/gzip.html) (.gz).

The original MNIST dataset is by Yann LeCun and Corinna Cortes and can be found [here](https://yann.lecun.org/exdb/mnist/). It is licensed under the Creative Commons Attribution-Share Alike 3.0 ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)) license.

The dataset license can be found in [here](data/LICENSE-DATA)

from collections import namedtuple

MNIST = namedtuple('MNIST', ['N', 'name', 'rows', 'cols', 'labels', 'images',
                             'kmeans', 'PCA', 'binarized'])

MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
TEST = ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

import numpy as np

class MNIST:
    def __init__(self, N, name, rows, cols, labels, images, kmeans, PCA, binarized):
        self.N = N
        self.name = name
        self.rows = rows
        self.cols = cols
        self.labels = labels
        self.images = images
        self.kmeans = kmeans
        self.PCA = PCA
        self.binarized = binarized

        self.i = 0
        self.epochs = 0

    def next_batch(self, batch_size):

        start = self.i
        self.i += batch_size

        if self.i > self.N:
            self.epochs += 1

            perm = np.arange(self.N)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]

            start = 0
            self.i = batch_size

        assert batch_size <= self.N
        end = self.i

        return self.images[start:end], self.labels[start:end]


MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
TEST = ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

def set_mnist(obj, attr, val):

    if not hasattr(obj, attr):
        raise Exception("No attributes named %s" % attr)

    setattr(obj, attr, val)

    return obj

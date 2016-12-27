import array
import random
import os.path
import struct
import gzip
import urllib.request

from src.kmeans import generate_kmeans
from src.pca import generate_pca
from src.const import *

import numpy as np
import matplotlib.pyplot as plt


def maybe_download(url, *files, data_dir='./data'):
    """Download a file if not present in the data dir"""
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    def maybe_extract(filename):
        if os.path.exists(name) or not filename:
            return

        f = open(name, 'wb')
        f.write(gzip.GzipFile(filename).read())
        f.close()

    def maybe_dl(f):
        filename = None
        if not os.path.exists(name):
            print('Attempting to download:', f)
            filename, _ = urllib.request.urlretrieve(url + f)

        return filename

    for f in files:
        name = os.path.join(data_dir, f[:-3])

        filename = maybe_dl(f)
        maybe_extract(filename)


def load_mnist(dataset='test', data_dir='./data', asbytes=True):
    """
    dataset: string
        The dataset to load :(test, train)
    data_dir: string
        the directory where to find images and labels
    asbytes : bool
        returns data as a numpy.uint8 instead of a numpy.float64

    Returns
    -------
    images : ndarray (N, rows * cols)
    labels : ndarray (N)
    """

    files = {'test': TEST, 'train': TRAIN}
    dtype = {True: np.uint8, False: np.float64}[asbytes]

    # The files are assumed to have these names and should be found in 'path'
    if not os.path.exists(data_dir):
        raise ValueError('Directory %s does not exist')

    try:
        p_images = os.path.join(data_dir, files[dataset][0])
        p_labels = os.path.join(data_dir, files[dataset][1])
    except KeyError:
        raise ValueError('dataset %s should be `test` or `train`' % dataset)

    for f in (p_images, p_labels):
        if not os.path.exists(f):
            raise ValueError('%s does not exist' % f)

    N, labels, images = 0, None, None

    with open(p_labels, 'rb') as f_labels:
        f_labels.seek(8)
        labels = np.array(array.array("b", f_labels.read()))

    with open(p_images, 'rb') as f_images:
        _, N, rows, cols = struct.unpack(">IIII", f_images.read(16))
        images = np.array(array.array("b", f_images.read()), dtype=dtype)
        images = np.reshape(images, (N, rows * cols))

    return MNIST(N, dataset, rows, cols, labels, images, None, None,
                 np.where(images > 0.5, 1, 0))


# Loads mnist and adds Kmeans, PCA, binarisation
def load_full_mnist(dataset):
    try:
        s = {'test': TEST, 'train': TRAIN}[dataset]
    except KeyError:
        raise ValueError('%s is not a valid dataset' % dataset)

    maybe_download(MNIST_URL, *(f + '.gz' for f in s))

    dataset = load_mnist(dataset)
    dataset = generate_kmeans(dataset)
    dataset = generate_pca(dataset)

    return dataset


def display_mnist_samples(dataset):
    for i in range(9):
        k = random.sample(np.where(dataset.labels == i)[0].tolist(), 1)[0]

        plt.subplot(int('33%s' % (i + 1)))
        plt.imshow(np.reshape(dataset.images[k], (dataset.rows, dataset.cols)), cmap='gray')
        plt.grid(True)

    plt.show()


train_dataset = load_full_mnist('train')
test_dataset = load_full_mnist('test')

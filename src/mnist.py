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
    print('dtype:', dtype)

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
        images = np.zeros((N, rows*cols))

        for i in range(0, N):
            n_pixels = rows*cols
            for j in range(0, n_pixels):
                pixel = f_images.read(1)
                value = pixel[0]
                images[i, j] = value

        return MNIST(N, dataset, rows, cols, labels, images, None, None,
                     np.where(images > 0.5, 1, 0))

    '''
    with open(p_images, 'rb') as f_images:
        _, N, rows, cols = struct.unpack(">IIII", f_images.read(16))
        images = np.array(array.array("b", f_images.read()), dtype=np.dtype('b'))
        images = np.reshape(images, (N, rows * cols))

    return MNIST(N, dataset, rows, cols, labels, images, None, None,
                 np.where(images > 0.5, 1, 0))
    '''


# Loads mnist and adds Kmeans, PCA, binarisation
def load_full_mnist(dataset, clusters, components):
    try:
        s = {'test': TEST, 'train': TRAIN}[dataset]
    except KeyError:
        raise ValueError('%s is not a valid dataset' % dataset)

    maybe_download(MNIST_URL, *(f + '.gz' for f in s))

    dataset = load_mnist(dataset, asbytes=False)
    #dataset = generate_kmeans(dataset, clusters)
    #dataset = generate_pca(dataset, components)

    return dataset


def display_mnist_samples(dataset):
    for i in range(9):
        k = random.sample(np.where(dataset.labels == i)[0].tolist(), 1)[0]

        plt.subplot(int('33%s' % (i + 1)))
        plt.imshow(np.reshape(dataset.images[k], (dataset.rows, dataset.cols)), cmap='gray')
        plt.grid(True)

    plt.show()


import pdb
def mk_valid_set(tr, clusters, components):
    N = 10000
    labels_idx = np.zeros(N, dtype='int32')
    vlabels = np.zeros(N)

    for i in range(10):
        idx = np.where(tr.labels == i)[0]
        np.random.shuffle(idx)
        idx = idx[:1000]

        labels_idx[(i * 1000):((i + 1) * 1000)] = idx
        vlabels[(i * 1000):((i + 1) * 1000)] = i

    vimages = tr.images[labels_idx]

    dataset = MNIST(N, 'validation', tr.rows, tr.cols, vlabels, vimages,
                    None, None, np.where(vimages > 0.5, 1, 0))

    #dataset = generate_kmeans(dataset, clusters)
    #dataset = generate_pca(dataset, components)


    tr = set_mnist(tr, 'N', tr.N - N)
    tr = set_mnist(tr, 'labels', np.delete(tr.labels, labels_idx))
    tr = set_mnist(tr, 'images', np.delete(tr.images, labels_idx, axis=0))

    return dataset, tr


def get_mnist_full(clusters=10, components=40):
    train_dataset = load_full_mnist('train', clusters, components)
    test_dataset = load_full_mnist('test', clusters, components)

    validation, train_dataset = mk_valid_set(train_dataset, clusters, components)

    return train_dataset, test_dataset, validation

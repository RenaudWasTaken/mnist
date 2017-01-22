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
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(dataset='test', data_dir='./data', asbytes=True):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    try:
        d = {'test': mnist.test,
             'train': mnist.train,
             'validation': mnist.validation}[dataset]
    except KeyError:
        raise ValueError('%s is not a valid dataset' % dataset)

    images = d.images
    labels = np.argmax(d.labels, axis=1)
    N = d.labels.shape[0]
    labels_2d = d.labels

    return MNIST(N, dataset, labels, labels_2d, images, d, None, None,
                 np.where(images > 0.5, 1, 0))

# Loads mnist and adds Kmeans, PCA, binarisation
def load_full_mnist(dataset, clusters, components):
    dataset = load_mnist(dataset, asbytes=False)
    print(dataset)
    if clusters:
        dataset = generate_kmeans(dataset, clusters)

    if components:
        dataset = generate_pca(dataset, components)

    return dataset


def get_mnist_full(clusters=None, components=None, validation=True):
    train_dataset = load_full_mnist('train', clusters, components)
    test_dataset = load_full_mnist('test', clusters, components)

    if validation:
        validation = load_mnist('validation', clusters, components)
        return train_dataset, test_dataset, validation

    return train_dataset, test_dataset

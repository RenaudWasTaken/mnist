import os.path
import math
import random

from sklearn.decomposition import PCA
from sklearn.externals import joblib
from src.const import *

import numpy as np
import matplotlib.pyplot as plt

def generate_pca(dataset, components=64, data_dir='./data'):
    d = dataset

    pca_dir = os.path.join(data_dir, 'pca/')
    p_dataset = os.path.join(pca_dir, d.name + '-%s.pkl' % components)

    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)

    if os.path.exists(p_dataset):
        print('Loading PCA from file %s for dataset %s' % (p_dataset, d.name))
        pca = joblib.load(p_dataset)
    else:
        print('Launching PCA')
        pca = PCA(components).fit(d.images)
        joblib.dump(pca, p_dataset, compress=1)

    return set_mnist(d, 'PCA', pca)


def display_pca_samples(dataset):
    d = dataset
    for i in range(9):
        k = random.sample(np.where(d.labels == i)[0].tolist(), 1)[0]
        s = int(math.sqrt(d.PCA.n_components))
        img = np.reshape(d.PCA.transform(d.images[k].reshape(1, -1)), (s, s))

        plt.subplot(int('33%s' % (i + 1)))
        plt.imshow(img)
        plt.grid(True)

    plt.show()


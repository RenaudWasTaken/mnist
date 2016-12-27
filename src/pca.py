import os.path
import math
import random

from sklearn.decomposition import PCA
from sklearn.externals import joblib
from src.const import MNIST

import numpy as np
import matplotlib.pyplot as plt

def generate_pca(dataset, components=64, data_dir='./data'):
    p_dataset = os.path.join(data_dir, dataset.name + '-pca.pkl')

    if os.path.exists(p_dataset):
        pca = joblib.load(p_dataset)
    else:
        print('Launching PCA')
        pca = PCA(components).fit(dataset.images)
        joblib.dump(pca, p_dataset, compress=1)

    d = dataset
    return MNIST(d.N, d.name, d.rows, d.cols, d.labels, d.images, d.kmeans, pca, d.binarized)


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


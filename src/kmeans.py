import os.path

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from src.const import MNIST

import numpy as np


def generate_kmeans(dataset, clusters=64, data_dir='./data'):
    p_dataset = os.path.join(data_dir, dataset.name + '-kmeans.pkl')

    if os.path.exists(p_dataset):
        kmeans = joblib.load(p_dataset)
    else:
        print('Launching Kmeans')
        kmeans = KMeans(clusters, verbose=1).fit(dataset.images)
        joblib.dump(kmeans, p_dataset, compress=1)

    d = dataset
    return MNIST(d.N, d.name, d.rows, d.cols, d.labels, d.images, kmeans, d.PCA, d.binarized)

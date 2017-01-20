import os
import os.path

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from src.const import MNIST


def generate_kmeans(dataset, clusters=64, data_dir='./data'):
    d = dataset

    kmeans_dir = os.path.join(data_dir, 'kmeans/')
    p_dataset = os.path.join(kmeans_dir, d.name + '-%s.pkl' % clusters)

    if not os.path.exists(kmeans_dir):
        os.makedirs(kmeans_dir)

    if os.path.exists(p_dataset):
        print('Loading kmeans from file %s for dataset %s' % (p_dataset, d.name))
        kmeans = joblib.load(p_dataset)
    else:
        print('Launching Kmeans')
        kmeans = KMeans(clusters, verbose=1).fit(d.images)
        joblib.dump(kmeans, p_dataset, compress=1)

    return MNIST(d.N, d.name, d.rows, d.cols, d.labels, d.images, kmeans, d.PCA, d.binarized)

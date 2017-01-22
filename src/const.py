import numpy as np
from collections import namedtuple as ndt

class MNIST(ndt('MNIST', ['N', 'name', 'labels', 'labels_2d', 'images', 'tf', 'kmeans', 'PCA',
                          'binarized'])):

    def __new__(self, N, name, labels, labels_2d, images, tf, kmeans, PCA, binarized):
        self = super(MNIST, self).__new__(self, N, name, labels, labels_2d, images,
                                          tf, kmeans, PCA, binarized)

        return self

    def __repr__(self):
        s = 'MNIST(N=%s, dataset="%s", labels=%s, images=%s' \
            % (self.N, self.name, self.labels.shape, self.images.shape)

        s += ', '

        if self.PCA:
            s += 'PCA=(%s components)' % self.PCA.components_
        else:
            s += 'PCA=False'

        s += ', '

        if self.kmeans:
            s += 'KMeans=(%s clusters)' % self.kmeans.cluster_centers_.shape[0]
        else:
            s += 'KMeans=False'

        s += ')'

        return s

    def next_batch(self, batch_size):
        return self.tf.next_batch(batch_size)

def set_mnist(obj, attr, val):

    if not hasattr(obj, attr):
        raise Exception("No attributes named %s" % attr)

    attrs = [v if k != attr else val for k, v in obj._asdict().items()]

    return MNIST(*attrs)


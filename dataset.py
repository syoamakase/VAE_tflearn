#!/usr/bin/env python
# -*-coding: utf-8 -*-

import pickle
import numpy as np
from numpy.random import *

class Datasets(object):
    def __init__(self):
        self.train = None
        self.test = None
        self.validation = None

    def load_data(self):
        return self.train.features, self.train.labels, self.test.features, self.test.labels

class Dataset(object):
    def __init__(self, features, labels):
        assert features.shape[0] == labels.shape[0], ("features.shape: %s labels.shape: %s" % (features.shape, labels.shape))
        self._num_examples = features.shape[0]

        features = features.astype(np.float32)
        features = np.multiply(features - 130.0, 1.0 / 70.0) # [130.0 - 200.0] -> [0 - 1]
        for i in range(features.shape[1]):
            feature = np.copy(features[:,i])
            std_feature = (feature - feature.mean()) / feature.std()
            features[:,i] = std_feature
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]


def sampling(n, mu1, sigma1, pi1, mu2, sigma2, pi2):
    norm_pi1 = pi1 / (pi1 + pi2)
    norm_pi2 = pi2 / (pi1 + pi2)
    mixiture_gauss = norm_pi1 * np.random.normal(mu1, sigma1,n) + norm_pi2 * np.random.normal(mu2, sigma2, n)
    data = np.reshape(mixiture_gauss, (n,1))
    return data


# shuffle data and its label in association
def corresponding_shuffle(data, target):
    random_indices = permutation(len(data))
    _data = np.zeros_like(data)
    _target = np.zeros_like(target)
    for i, j in enumerate(random_indices):
        _data[i] = data[j]
        _target[i] = target[j]
    return _data, _target


if __name__ == "__main__":
    datasets = Datasets()

    n = 500 # number of data
    x_class0 = sampling(n, 0, 0.5, 0.5, 2, 0.5, 0.5)
    x_class1 = sampling(n, 1, 0.5, 0.5, 3, 0.5, 0.5)
    y_class0 = np.zeros_like(x_class0) # class0 :0
    y_class1 = np.ones_like(x_class1) # class1 :1

    # concat
    data = np.r_[x_class0, x_class1]
    target = np.r_[y_class0, y_class1]

    # shuffle
    data, target = corresponding_shuffle(data, target)

    # split data
    N_train = np.floor(n * 2 * 0.8).astype(np.int32)
    N_validation = np.floor(N_train * 0.2).astype(np.int32)
    x_train, x_test = np.split(data, [N_train])
    y_train, y_test = np.split(target, [N_train])
    x_validation = x_train[:N_validation]
    y_validation = y_train[:N_validation]

    # create dataset
    datasets.train = Dataset(x_train, y_train)
    datasets.test = Dataset(x_test, y_test)
    datasets.validation = Dataset(x_validation, y_validation)

    # save a pickle file
    with open('mixture.pkl', 'wb') as f:
        pickle.dump(datasets, f)

    print("complete!!")
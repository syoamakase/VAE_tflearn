#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import range

import tensorflow as tensorflow
import tflearn
import vae

from tflearn.datasets import mnist
import numpy as np

batch_size = 128
latent_dim = 2
hidden_dim = 512

X, Y, testX, testY = mnist.load_data()

print(X.shape)
original = X.shape[1:]
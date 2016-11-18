#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import range
import numpy as np

import tensorflow as tf
import tflearn
from tflearn.datasets import mnist

import vae

TENSORBOARD_DIR = 'tensor_logs/'

batch_size = 128
latent_dim = 2
hidden_dim = 512

X, Y, testX, testY = mnist.load_data()

original_shape = X.shape[1:]
original_shape = [original_shape[i] for i in range(len(original_shape))]
input_shape = [None] + original_shape

x = tflearn.input_data(shape=input_shape)
z, mu, log_var = vae.encode(x, hidden_dim=hidden_dim, latent_dim=latent_dim)

x_decoded_mean = vae.decode(z, hidden_dim=hidden_dim, original_shape=original_shape)

def loss_vae(x_hat, x):
    with tf.variable_op_scope([x_hat, x], 'vae_loss') as scope:
        bce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat,x), reduction_indices=1)
        kl_loss = - 0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var), reduction_indices=1)
        loss = tf.reduce_mean(bce_loss + kl_loss)

    return loss


vae = tflearn.regression(x_decoded_mean, optimizer='adam',loss=loss_vae, metric=None)
vae = tflearn.DNN(vae, tensorboard_verbose=0, tensorboard_dir=TENSORBOARD_DIR,
					checkpoint_path='model_variational_autoencoder',max_checkpoints=1)
vae.fit(X, X, n_epoch=2, batch_size=batch_size, run_id='varational_auto_encoder')


#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import range

import tensorflow as tf
import tflearn


def encode(input_data, hidden_dim, latent_dim=None):
    with tf.variable_op_scope([input_data], 'encoder') as scope:
        
        encoder = tflearn.fully_connected(input_data, hidden_dim, activation='relu')
        
        encoder = tflearn.fully_connected(encoder, hidden_dim, activation='relu')

        mean = tflearn.fully_connected(encoder, latent_dim)
        log_var = tflearn.fully_connected(encoder, latend_dim)
        std = tf.exp(0.5 * log_var)
        random_sample = tf.random_normal(tf.shape(log_var))
        z = tf.add(mean, tf.mul(std, random_sample))

    return z, mean, log_var


def decode(input_data, hidden_dim=None, original=None):
    with tf.variable_op_scope([input_data, 'decoder']) as scope:
        decoder = tf.fully_connected(input_data, hidden_dim, activation='relu')

        decoder = tflearn.fully_connected(decoder, hidden_dim, activation='relu')

        mean = tflearn.fully_connected(decoder, original[0], activation='sigmoid')

    return mean

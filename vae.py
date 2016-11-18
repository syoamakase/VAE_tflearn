#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import range

import tensorflow as tf
import tflearn


def encode(input_data, hidden_dim, latent_dim=None):
    with tf.variable_op_scope([input_data], 'encoder'):
        
        encoder = tflearn.fully_connected(input_data, hidden_dim, activation='relu') 
        encoder = tflearn.fully_connected(encoder, hidden_dim, activation='relu')

        mu = tflearn.fully_connected(encoder, latent_dim, name='network_mu')
        log_var = tflearn.fully_connected(encoder, latent_dim, name='network_logvar')
        std = tf.exp(0.5 * log_var)
        random_sample = tf.random_normal(tf.shape(log_var))
        z = tf.add(mu, tf.mul(std, random_sample))

    return z, mu, log_var


def decode(input_data, hidden_dim=None, original_shape=None):
    with tf.variable_op_scope([input_data], 'decoder'):
        decoder = tflearn.fully_connected(input_data, hidden_dim, activation='relu', name='latent_input')
        decoder = tflearn.fully_connected(decoder, hidden_dim, activation='relu',name='decoder_hidden1')

        mean = tflearn.fully_connected(decoder, original_shape[0], activation='sigmoid', name='original_hat')

    return mean

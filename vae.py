#!/usr/bin/env python
# -*- coding: utf-8 -*-

from six.moves import range

import tensorflow as tf
import tflearn

class Latent():
	def __init__(self):
		self.mu = None
		self.log_var = None
		self.pi = None
		self.std = None


def encode(input_data, hidden_dim, latent_dim=None, mixture_num=1):
    with tf.variable_op_scope([input_data], 'encoder'):
        
        encoder = tflearn.fully_connected(input_data, hidden_dim, activation='relu') 
        encoder = tflearn.fully_connected(encoder, hidden_dim, activation='relu')

        latents = []
        for mn in range(mixture_num):
        	latent = Latent()
        	latent.mu = tflearn.fully_connected(encoder, latent_dim, name='network_mu'+str(mn))
        	latent.log_var = tflearn.fully_connected(encoder, latent_dim, name='network_logvar'+str(mn))
        	latent.pi = tflearn.fully_connected(encoder, latent_dim, name='network_pi'+str(mn))
        	latent.std = tf.exp(0.5 * latent.log_var)
        	latents.append(latent)

        print(latents)
        random_sample = tf.random_normal(tf.shape(latents[0].log_var))
        # pi_weight1 = pi1 / (pi1 + pi2) 
        # pi_weight2 = pi2 / (pi1 + pi2)
        # z = tf.add(tf.add(pi_weight1 * mu1,  pi_weight2 * mu2), tf.mul(tf.add(pi_weight1 * std1, pi_weight2 * std2), random_sample))
        z = tf.add(latents[0].mu, tf.mul(latents[0].std, random_sample))

    return z, latent


def decode(input_data, hidden_dim=None, original_shape=None):
    with tf.variable_op_scope([input_data], 'decoder'):
        decoder = tflearn.fully_connected(input_data, hidden_dim, activation='relu', name='latent_input')
        decoder = tflearn.fully_connected(decoder, hidden_dim, activation='relu',name='decoder_hidden1')

        mean = tflearn.fully_connected(decoder, original_shape[0], activation='sigmoid', name='original_hat')

    return mean

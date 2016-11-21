from six.moves import range

import tensorflow as tf
import tflearn
from tflearn.datasets import mnist
import vae
from vae import Latent

import numpy as np
from skimage import io

original_dim = 784
latent_dim = 2
hidden_dim = 512
mixture_num = 2
model_file = 'model_vae-21500'

X, Y, testX, testY = mnist.load_data()

original_shape = X.shape[1:]
original_shape = [original_shape[i] for i in range(len(original_shape))]

with tf.Graph().as_default():
    input_shape = [None] + original_shape
    x = tflearn.input_data(shape=input_shape)
    z, latents = vae.encode(x, hidden_dim=hidden_dim,
                                latent_dim=latent_dim,mixture_num=mixture_num)
    encoder = tflearn.DNN(z)
    optargs = {'scope_for_restore':'encoder'}
    encoder.load(model_file, optargs)
    mu_encoder0 = tflearn.DNN(latents[0].mu)
    mu_encoder0.load(model_file, optargs)
    log_var_encoder0 = tflearn.DNN(latents[0].log_var)
    log_var_encoder0.load(model_file, optargs)
    pi_encoder0 = tflearn.DNN(latents[0].pi)
    pi_encoder0.load(model_file, optargs)
    if(mixture_num == 2):
        mu_encoder1 = tflearn.DNN(latents[1].mu)
        mu_encoder1.load(model_file, optargs)
        log_var_encoder1 = tflearn.DNN(latents[1].log_var)
        log_var_encoder1.load(model_file, optargs)
        pi_encoder1 = tflearn.DNN(latents[1].pi)
        pi_encoder1.load(model_file, optargs)

with tf.Graph().as_default():
    # build a digit generator that can sample from the learned distribution
    decoder_input = tflearn.input_data(shape=[None, latent_dim])
    gen_decoded_mean = vae.decode(decoder_input, hidden_dim=hidden_dim,
                                  original_shape=original_shape)
    generator = tflearn.DNN(gen_decoded_mean)
    generator.load(model_file, {'scope_for_restore':'decoder'})

digit_size = 28
n = 15
linspace = 1000
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-linspace, linspace, n)
grid_y = np.linspace(-linspace, linspace, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi] + [0 for k in range(2, latent_dim)]])
        x_decoded = generator.predict(z_sample)
        digit = np.reshape(x_decoded[0], [digit_size, digit_size])
        figure[i * digit_size : (i + 1) * digit_size,
               j * digit_size : (j + 1) * digit_size] = digit
figure *= 255
figure = figure.astype(np.uint8)

io.imsave('vae_z.png', figure)

figure = np.ndarray(shape=(digit_size * (n), digit_size * (n)),
                    dtype=np.float16)
testX = tflearn.data_utils.shuffle(X)[0][0:1]

if(mixture_num == 1):
    testMu = mu_encoder0.predict(testX)[0]
    testLogVar = log_var_encoder0.predict(testX)[0]
else:
    norm_pi0 = np.array(pi_encoder0.predict(testX)[0]) /(np.array(pi_encoder0.predict(testX)[0]) + np.array(pi_encoder1.predict(testX)[0]))
    norm_pi1 = np.array(pi_encoder1.predict(testX)[0]) /(np.array(pi_encoder0.predict(testX)[0]) + np.array(pi_encoder1.predict(testX)[0]))
    mu0 = np.array(mu_encoder0.predict(testX)[0]) * norm_pi0
    mu1 = np.array(mu_encoder1.predict(testX)[0]) * norm_pi1
    testMu = mu0 + mu1
    log_var0 = np.array(log_var_encoder0.predict(testX)[0]) * norm_pi0
    log_var1 = np.array(log_var_encoder1.predict(testX)[0]) * norm_pi1
    testLogVar = log_var0 + log_var1

std = [np.exp(0.5 * testLogVar[i]) * 4 for i in range(2)]
grid_x = np.linspace(-std[0], std[0], n) + testMu[0]
grid_y = np.linspace(-std[1], std[1], n) + testMu[1]
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi] + [testMu[k] for k in range(2, latent_dim)]])
        x_decoded = generator.predict(z_sample)
        digit = np.reshape(x_decoded[0], [digit_size, digit_size])
        figure[i * digit_size : (i + 1) * digit_size,
               j * digit_size : (j + 1) * digit_size] = digit
figure *= 255
figure = figure.astype(np.uint8)

io.imsave('vae_std.png', figure)
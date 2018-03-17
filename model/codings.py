import tensorflow as tf

from tensorflow.contrib import layers
from utils.activations import leaky_rectify


def gan_conv_anime_discriminator(x, activation=None):
    """
    Inference network q(z|x) which encodes a mini batch of data points
    to a parameterization of a diagonal Gaussian using a network with
    convolutional layers.
    :param x: Mini batch of data points to encode.
    :param latent_dim: dimension of latent space into which we encode
    :return: e: Encoded mini batch.
    """
    normalizer_params = {'decay': 0.9, 'scale': True, 'center': False}
    e = tf.reshape(x, [-1, 64, 64, 3])
    filter_size = 64
    e = layers.conv2d(e, filter_size, 5,
                      activation_fn=leaky_rectify,
                      padding='SAME', stride=2, scope='conv-0{0}'.format(0))

    for i in range(2):
        filter_size = filter_size*2
        e = layers.conv2d(e, filter_size, 5,
                          normalizer_fn=tf.contrib.layers.batch_norm,
                          normalizer_params=normalizer_params,
                          activation_fn=leaky_rectify,
                          padding='SAME', stride=2, scope='conv-0{0}'.format(i + 1))
    e = layers.conv2d(e, 32, 3,
                      activation_fn=leaky_rectify,
                      normalizer_fn=tf.contrib.layers.batch_norm,
                      normalizer_params=normalizer_params,
                      padding='SAME', stride=1, scope='conv-final')
    e = layers.flatten(e)

    e = layers.fully_connected(e, 128,
                               activation_fn=leaky_rectify,
                               normalizer_fn=tf.contrib.layers.batch_norm,
                               normalizer_params=normalizer_params,
                               scope='fc-01')
    output = layers.fully_connected(e, 1,
                                    scope='fc-final', activation_fn=activation)

    return output


def gan_conv_anime_generator(z, z_size):

    filter_size = 512
    normalizer_params = {'decay': 0.9, 'scale': True, 'center': False}

    x = layers.fully_connected(z, 8192,
                               normalizer_fn=tf.contrib.layers.batch_norm,
                               normalizer_params=normalizer_params,
                               scope='fc-01', activation_fn=None)
    feature = x
    x = tf.reshape(x, [-1, 4, 4, filter_size])

    for i in range(4):
        filter_size = filter_size / 2
        x = layers.conv2d_transpose(x, filter_size, 5,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params=normalizer_params,
                                    padding='SAME', stride=2, scope='conv-transpose-0{0}'.format(i + 1))
    x = layers.conv2d_transpose(x, 3, 3,
                                padding='SAME', stride=1, scope='conv-transpose-final', activation_fn=None)
    image = tf.tanh(layers.flatten(x))

    z_prediction = layers.fully_connected(feature, z_size,
                                          scope='fc-bk-01', activation_fn=None)

    return image, z_prediction
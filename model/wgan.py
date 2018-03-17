import re
import numpy as np
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

from utils.activations import lrelu,leaky_rectify


class GAN(object):
    def __init__(self, latent_dim, observation_dim, batch_size, generator, discriminator):

        self._latent_dim = latent_dim
        self._observation_dim = observation_dim
        self._batch_size = batch_size
        self._generator = generator
        self._discriminator = discriminator
        self.clip_values = (-0.02, 0.02)
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('gan'):
            self.x = tf.placeholder(tf.float32, shape=[None, self._observation_dim])
            self.z = tf.placeholder(tf.float32, shape=[self._batch_size, self._latent_dim])
            with tf.variable_scope('generator'):
                    self.generated, z_prediction = self._generator(self.z, self._latent_dim)

            with tf.variable_scope('discriminator'):
                    self.fake = self._discriminator(self.generated, None)
            with tf.variable_scope('discriminator', reuse=True):
                    self.real = self._discriminator(self.x, None)

            with tf.variable_scope('discriminator', reuse=True):
                    alpha = tf.random_uniform(
                        shape=[self._batch_size, 1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = self.generated - self.x
                    interpolates = self.x + (alpha * differences)
                    gradients = tf.gradients(self._discriminator(interpolates, None), [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                    self.gradient_penalty = gradient_penalty*0.1


            with tf.variable_scope('loss'):
                with tf.variable_scope('d-loss'):
                    self.d_loss =  tf.reduce_mean(self.fake - self.real) + self.gradient_penalty

                with tf.variable_scope('g-loss'):
                    self.g_loss = -tf.reduce_mean(self.fake)

                with tf.variable_scope('f-loss'):
                    self.f_loss = tf.reduce_mean(tf.square(self.z - z_prediction))


            with tf.variable_scope('optimizer'):
                with tf.variable_scope('d-solver'):
                    discriminator_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gan/discriminator")
                    self.d_solver = (tf.train
                                     .RMSPropOptimizer(5e-5)
                                     .minimize(self.d_loss, var_list=discriminator_var))

                with tf.variable_scope('g-solver'):
                    generator_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gan/generator")
                    self.g_solver = (tf.train
                                     .RMSPropOptimizer(5e-5)
                                     .minimize(self.g_loss, var_list=generator_var))

                with tf.variable_scope('f-solver'):
                    self.f_solver = (tf.train
                                     .RMSPropOptimizer(5e-5)
                                     .minimize(self.f_loss, var_list=generator_var))

            with tf.variable_scope('clip'):
                self.clip = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1]))
                                             for var in discriminator_var]

            # start tensorflow session
            self._sesh = tf.Session()
            init = tf.global_variables_initializer()
            self._sesh.run(init)

    def update(self, x, epoch):


        z = np.random.normal(size=[self._batch_size, self._latent_dim])
        d_loss_curr  = self._sesh.run(self.d_loss, feed_dict={self.x: x, self.z: z})
        self._sesh.run(self.d_solver, feed_dict={self.x: x, self.z: z})
        self._sesh.run(self.clip)

        z = np.random.normal(size=[self._batch_size, self._latent_dim])
        g_loss_curr = self._sesh.run(self.g_loss, feed_dict={self.z: z})
        self._sesh.run(self.g_solver, feed_dict={self.z: z})
        self._sesh.run(self.f_solver, feed_dict={self.z: z})
        return d_loss_curr, g_loss_curr

    def z2x(self, z, MNIST=True):

        x = self._sesh.run([self.generated], feed_dict={self.z: z})
        # need to reshape since our network processes batches of 1-D 28 * 28 arrays
        if MNIST:
            x = np.array(x)[:, 0, :].reshape(28, 28)
        else:
            x = x[0].reshape(-1, 64, 64, 3)
        return x

    def save_generator(self, path, prefix="is/generator"):
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "generator" in v.name:
                name = prefix + re.sub("gan/generator", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)
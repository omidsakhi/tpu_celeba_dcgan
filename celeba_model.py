"""Simple generator and discriminator models.
Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _dense(x, channels, name):
  return tf.layers.dense(
      x, channels,
      bias_initializer=None,
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _dense_with_bias(x, channels, name):
  return tf.layers.dense(
      x, channels,
      bias_initializer=tf.zeros_initializer(),      
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _conv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=None,
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _conv2d_with_bias(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=tf.zeros_initializer(),      
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _deconv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=None,
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _deconv2d_with_bias(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=tf.zeros_initializer(),      
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def discriminator(x, is_training=True, scope='Discriminator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    fn = 64

    x = _conv2d_with_bias(x, fn, 5, 2, name='d_conv1') # x: 64x64
    x = _leaky_relu(x)

    x = _conv2d(x, fn * 2, 5, 2, name='d_conv2') # x: 32x32
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

    x = _conv2d(x, fn * 4, 5, 2, name='d_conv3') # x: 16x16
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))

    x = _conv2d(x, fn * 4, 5, 2, name='d_conv4') # x: 8x8
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn4'))

    x = _conv2d(x, fn * 4, 5, 2, name='d_conv5') # x: 4x4
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn5'))

    x = tf.reshape(x, [-1, 4 * 4 * fn * 4])

    x = _dense_with_bias(x, 1, name='d_fc_6')

    return x


def generator(x, is_training=True, scope='Generator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    
    fn = 64

    x = _dense(x, fn * 4 * 4 * 4, name='g_fc1')
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn1'))

    x = tf.reshape(x, [-1, 4, 4, fn * 4]) #4x4

    x = _deconv2d(x, fn * 4, 3, 2, name='g_dconv2') #8x8
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn2'))

    x = _deconv2d(x, fn * 4, 3, 2, name='g_dconv3') #16x16
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn3'))

    x = _deconv2d(x, fn * 2, 3, 2, name='g_dconv4') #32x32
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn4'))

    x = _deconv2d(x, fn, 3, 2, name='g_dconv5') #64x64
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn5'))

    x = _deconv2d_with_bias(x, 3, 3, 2, name='g_dconv6') #128x128    

    x = tf.tanh(x)    

    return x
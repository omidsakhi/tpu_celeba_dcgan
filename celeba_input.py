from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image

def parser(serialized_example):  
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'labels': tf.FixedLenFeature([], tf.string),
      })    
  image = tf.image.decode_jpeg(features['image'])
  image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0  
  image = tf.reshape(image, [3, 128*128])
  labels = tf.constant(-1.0, shape=[40]) #tf.cast(features['labels'], tf.int32)  
  return image, labels


class InputFunction(object):  
  
  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim

  def __call__(self, params):      
    batch_size = params['batch_size']    
    data_dir = params['data_dir']    
    dataset = tf.data.TFRecordDataset([
        data_dir + '/data_0.tfrecords',
        data_dir + '/data_1.tfrecords',
        data_dir + '/data_2.tfrecords',
        data_dir + '/data_3.tfrecords',
        data_dir + '/data_4.tfrecords'
         ])
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(2)
    images, labels = dataset.make_one_shot_iterator().get_next()
    images = tf.reshape(images, [batch_size, 128, 128, 3])
    random_noise = tf.random_normal([batch_size, self.noise_dim])
    features = {
        'real_images': images,
        'random_noise': random_noise}

    return features, labels

def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  return img    
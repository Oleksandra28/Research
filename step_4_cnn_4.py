#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This is an example of using convolutional networks over characters
for DBpedia dataset to predict class from description of an entity.
This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626
and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""

from __future__ import absolute_import

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import skflow
import numpy as np
from cnn_utilities import *
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import *


import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
from tensorflow.contrib import learn

N_FEATURES = 140*26
N_FILTERS = 10
WINDOW_SIZE = 3


def my_conv_model(x, y):

    print("x ", tf.shape(x))
    print("y ", tf.shape(y))

    # to form a 4d tensor of shape batch_size x 1 x N_FEATURES x 1
    x = tf.reshape(x, [-1, 1, N_FEATURES, 1])

    # this will give sliding window of 1 x WINDOW_SIZE convolution.
    features = tf.contrib.layers.convolution2d(inputs=x,
                                               num_outputs=N_FILTERS,
                                               kernel_size=[1, WINDOW_SIZE],
                                               padding='VALID')

    print("features ", tf.shape(features)[0], tf.shape(features)[1], tf.shape(features)[2], tf.shape(features)[3]) #features  (?, 1, 3638, 10)
    print(features) # Tensor("Conv/Relu:0", shape=(?, 1, 3638, 10), dtype=float32)

    # Max pooling across output of Convolution+Relu.
    pool = tf.nn.max_pool(features, ksize=[1, 1, 2, 1],
                             strides=[1, 1, 2, 1], padding='SAME')

    pool_shape = tf.shape(pool)
    y_shape = tf.shape(y)
    print("pool_shape (1): ", pool_shape) #pool  (?, 1, 1819, 10)
    print("y_shape (1):    ", y_shape) #y  (?,)

    pool = tf.reshape(pool, [pool_shape[0], pool_shape[2]*pool_shape[3]])
    #y = tf.reshape(y, [y_shape[0], 1])
    y = tf.expand_dims(y, 1)

    pool_shape = tf.shape(pool)
    # y_shape = tf.shape(y)
    y = tf.expand_dims(y, 1)
    print("pool_shape (2): ", pool_shape) #pool  (?, 1, 1819, 10)
    print("y_shape (2):    ", y_shape) #y  (?,)

    prediction, loss = learn.models.logistic_regression(pool, y)
    return prediction, loss

"""
# N = 256
# FILTER_SHAPE1 = [20, N]
# FILTER_SHAPE2 = [20, N_FILTERS]
# POOLING_WINDOW = 4
# POOLING_STRIDE = 2
# def char_cnn_model(x, y):
#     ###Character level convolutional neural network model to predict classes.
#
#     x = tf.cast(x, tf.float32)
#     y = tf.cast(y, tf.float32)
#
#     print("x ", tf.shape(x)) #Tensor("Shape:0", shape=(2,), dtype=int32)
#     print("y ", tf.shape(y)) #Tensor("Shape_1:0", shape=(1,), dtype=int32)
#
#     byte_list = tf.reshape(x, [-1, N_FEATURES, 1, 1])
#     with tf.variable_scope('CNN_Layer1'):
#       # Apply Convolution filtering on input sequence.
#       conv1 = tf.contrib.layers.convolution2d(inputs=byte_list,
#                                               num_outputs=N_FILTERS,
#                                               kernel_size=FILTER_SHAPE1,
#                                               padding='VALID')
#       # Add a RELU for non linearity.
#       conv1 = tf.nn.relu(conv1)
#       # Max pooling across output of Convolution+Relu.
#       pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1],
#                              strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
#       # Transpose matrix so that n_filters from convolution becomes width.
#       pool1 = tf.transpose(pool1, [0, 1, 3, 2])
#     with tf.variable_scope('CNN_Layer2'):
#       # Second level of convolution filtering.
#       conv2 = tf.contrib.layers.convolution2d(inputs=pool1,
#                                               num_outputs=N_FILTERS,
#                                               kernel_size=FILTER_SHAPE2,
#                                               padding='VALID')
#       # Max across each filter to get useful features for classification.
#       pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
#
#     print (pool2.get_shape()) #(?, 10)
#     print (y.get_shape()) #(?,)
#     print (pool2) #(?, 10)
#     print (y) #(?,)
#     # Apply regular WX + B and classification.
#     prediction, loss = learn.models.logistic_regression(pool2, y)
#
#     train_op = tf.contrib.layers.optimize_loss(
#         loss, tf.contrib.framework.get_global_step(),
#         optimizer='Adam', learning_rate=0.01)
#
#     return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op
"""

def main(unused_argv):

    # training and testing data encoded as one-hot
    data_folder = './data'

    sandyData = np.loadtxt(data_folder+'/sandyData.csv', delimiter=',')
    sandyLabels = np.loadtxt(data_folder+'/sandyLabels.csv', delimiter=',')

    x_train, x_test, y_train, y_test = \
        train_test_split(sandyData, sandyLabels, test_size=0.2, random_state=7)

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Build model
    classifier = learn.Estimator(model_fn=my_conv_model)

    # Train and predict
    classifier.fit(x_train, y_train, steps=100)
    y_predicted = [p['class'] for p in classifier.predict(x_test, as_iterable=True)]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()
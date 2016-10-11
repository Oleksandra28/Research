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


import numpy as np
from sklearn import metrics
import traceback

import tensorflow as tf
from tensorflow.contrib import learn
import sys

N_FEATURES = 140*26
N_FILTERS = 10
WINDOW_SIZE = 3


def my_conv_model(x, y):

    # 1. form a 4d tensor of shape N x 1 x N_FEATURES x 1
    x = tf.reshape(x, [-1, 1, N_FEATURES, 1])

    # 2. this will give sliding window of 1 x WINDOW_SIZE convolution.
    # kernel_size - size of a sliding window
    conv1 = tf.contrib.layers.convolution2d(inputs=x,
                                               num_outputs=N_FILTERS,
                                               kernel_size=[1, WINDOW_SIZE],
                                               stride=[1,1],
                                               padding='VALID')

    # 3. Add a RELU for non linearity.
    conv1 = tf.nn.relu(conv1)

    # 4. Max pooling across output of Convolution+Relu.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 2, 1],
                             strides=[1, 1, 2, 1], padding='SAME')

    print("(1) pool_shape", pool1.get_shape()) #pool  (?, 1, 1819, 10)
    print("(1) y_shape", y.get_shape()) #y  (?,)

    pool_shape = pool1.get_shape()
    #pool1 = tf.reshape(pool1, [-1, (pool_shape[2] * pool_shape[3]).value])

    y = tf.expand_dims(y, 1)

    print("(2) pool_shape", pool1.get_shape()) #pool  (?, 1, 1819, 10)
    print("(2) y_shape", y.get_shape()) #y  (?,)

    # Second level of convolution filtering.
    conv2 = tf.contrib.layers.convolution2d(inputs=pool1,
                                            num_outputs=N_FILTERS,
                                            kernel_size=[1, WINDOW_SIZE],
                                            padding='VALID')
    # Max across each filter to get useful features for classification.
    #pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 2, 1],
                             strides=[1, 1, 2, 1], padding='SAME')
    pool_shape = pool2.get_shape()
    pool2 = tf.reshape(pool2, [-1, (pool_shape[2] * pool_shape[3]).value])

    try:
        exc_info = sys.exc_info()

        print("(3) pool_shape", pool2.get_shape())
        print("(3) y_shape", y.get_shape())

        prediction, loss = learn.models.logistic_regression(pool2, y)
        train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=tf.contrib.framework.get_global_step(),
                    optimizer='SGD',
                    learning_rate=0.001)
        print("====================================================")

        return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

    except Exception:
        #print(traceback.format_exc())
        pass
    finally:
        # Display the *original* exception
        #traceback.print_exception(*exc_info)
        #del exc_info
        pass

def main(unused_argv):

    global N

    # training and testing data encoded as one-hot
    data_folder = './data'

    sandyData = np.loadtxt(data_folder+'/sandyData.csv', delimiter=',')
    sandyLabels = np.loadtxt(data_folder+'/sandyLabels.csv', delimiter=',')

    x_train, x_test, y_train, y_test = \
        train_test_split(sandyData, sandyLabels, test_size=0.2)#, random_state=3)

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print(x_train.shape) #(7196, 3640)
    print(y_train.shape)

    #N = x_train.shape[0]
    N = 100

    x_train = x_train[:N,:]
    y_train = y_train[:N]
    print(x_train.shape)
    print(y_train.shape)

    # x_test = x_test[:N,:]
    # y_test = y_test[:N]

    print(x_test.shape)

    # Build model
    classifier = learn.Estimator(model_fn=my_conv_model)

    # Train and predict
    classifier.fit(x_train, y_train, steps=100)

    N = y_test.shape[0]

    print("---after fitting---")
    y_predicted = [p['class'] for p in classifier.predict(x_test, as_iterable=True)]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()
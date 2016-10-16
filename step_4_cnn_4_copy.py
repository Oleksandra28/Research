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

from utilities import *

import numpy as np
from sklearn import metrics
import traceback

import tensorflow as tf
from tensorflow.contrib import learn
import sys

N_FEATURES = 140*27
N_FILTERS = 10
WINDOW_SIZE = 3


def my_conv_model(x, y):

    # 1. form a 4d tensor of shape N x 1 x N_FEATURES x 1
    x = tf.reshape(x, [-1, 1, N_FEATURES, 1])

    ##########################################################################
    ##### Conv layer 1 #####
    conv1 = tf.contrib.layers.convolution2d(inputs=x,
                                            num_outputs=N_FILTERS,
                                            kernel_size=[1, 7],
                                            stride=[1, 1],
                                            padding='VALID')

    # 3. Add a RELU for non linearity.
    conv1 = tf.nn.relu(conv1)

    # 4. Max pooling across output of Convolution+Relu.
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 3, 1],
                           padding='SAME')

    ##########################################################################
    ##### Conv layer 2 #####
    conv2 = tf.contrib.layers.convolution2d(inputs=pool1,
                                            num_outputs=N_FILTERS,
                                            kernel_size=[1, 7],
                                            padding='VALID')

    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 1, 2, 1],
                           strides=[1, 1, 2, 1],
                           padding='SAME')

    # ##########################################################################
    # ##### Conv layer 3 #####
    # conv3 = tf.contrib.layers.convolution2d(inputs=pool2,
    #                                         num_outputs=N_FILTERS,
    #                                         kernel_size=[1, 7],
    #                                         padding='VALID')
    #
    # pool3 = tf.nn.max_pool(conv3,
    #                        ksize=[1, 1, 2, 1],
    #                        strides=[1, 1, 2, 1],
    #                        padding='SAME')
    #
    # ##########################################################################
    # ##### Conv layer 4 #####
    # conv4 = tf.contrib.layers.convolution2d(inputs=pool3,
    #                                         num_outputs=N_FILTERS,
    #                                         kernel_size=[1, 7],
    #                                         padding='VALID')
    #
    # pool4 = tf.nn.max_pool(conv4,
    #                        ksize=[1, 1, 2, 1],
    #                        strides=[1, 1, 2, 1],
    #                        padding='SAME')
    #
    # ##########################################################################
    # ##### Conv layer 5 #####
    # conv5 = tf.contrib.layers.convolution2d(inputs=pool4,
    #                                         num_outputs=N_FILTERS,
    #                                         kernel_size=[1, 7],
    #                                         padding='VALID')
    #
    # pool5 = tf.nn.max_pool(conv5,
    #                        ksize=[1, 1, 2, 1],
    #                        strides=[1, 1, 2, 1],
    #                        padding='SAME')
    # ##########################################################################
    # ##### Fully connected layer 1 #####
    #
    # last_pool_layer = pool5
    # last_pool_layer_shape = last_pool_layer.get_shape()
    # n_cols = (last_pool_layer_shape[2] * last_pool_layer_shape[3]).value
    # last_pool_layer = tf.reshape(last_pool_layer, [-1, n_cols])
    #
    # fc_layer1 = tf.contrib.layers.fully_connected(inputs=last_pool_layer,
    #                                   num_outputs=10,
    #                                   activation_fn=tf.nn.relu)
    # # Apply Dropout
    # fc_layer1 = tf.nn.dropout(fc_layer1, keep_prob=0.5)
    #
    # ##########################################################################
    # ##### Fully connected layer 2 #####
    #
    # fc_layer2 = tf.contrib.layers.fully_connected(inputs=fc_layer1,
    #                                   num_outputs=10,
    #                                   activation_fn=tf.nn.relu)
    # # Apply Dropout
    # fc_layer2 = tf.nn.dropout(fc_layer2, keep_prob=0.5)
    #
    # ##########################################################################
    # ##### Fully connected layer 3 #####
    #
    # fc_layer3 = tf.contrib.layers.fully_connected(inputs=fc_layer2,
    #                                   num_outputs=10,
    #                                   activation_fn=tf.nn.relu)

    ##########################################################################
    ##### Fully connected layer #####

    last_pool_layer = pool2
    last_pool_layer_shape = last_pool_layer.get_shape()
    n_cols = (last_pool_layer_shape[2] * last_pool_layer_shape[3]).value
    last_pool_layer = tf.reshape(last_pool_layer, [-1, n_cols])
    fc_layer = tf.contrib.layers.fully_connected(inputs=pool2,
                                      num_outputs=10,
                                      activation_fn=tf.nn.relu)

    last_layer = fc_layer
    try:
        last_layer_shape = last_layer.get_shape()
        print("last_layer_shape", last_layer_shape)
        last_layer = tf.reshape(last_layer, [-1, (last_layer_shape[2] * last_layer_shape[3]).value])
        print("last_layer_shape", last_layer.get_shape())

        exc_info = sys.exc_info()

        y = tf.expand_dims(y, 1)

        prediction, loss = learn.models.logistic_regression(last_layer, y)
        print("prediction", prediction)
        prediction = tf.Print(prediction, [prediction], message="This is a: ")
        #print(prediction.eval())

        train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=tf.contrib.framework.get_global_step(),
                    optimizer='SGD',
                    learning_rate=0.001)

        #return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op
        return {'class': prediction, 'prob': prediction}, loss, train_op

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
        train_test_split(sandyData, sandyLabels, test_size=0.2, random_state=7)

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print(x_train.shape) #(7196, 3640)
    print(y_train.shape)

    #N = x_train.shape[0]
    N = 2000

    # x_train = x_train[:N,:]
    # y_train = y_train[:N]
    print("x_train.shape",x_train.shape)
    print("y_train.shape",y_train.shape)

    # x_test = x_test[:N,:]
    # y_test = y_test[:N]

    print("x_test.shape",x_test.shape)
    print("y_test.shape",y_test.shape)

    # Build model
    classifier = learn.Estimator(model_fn=my_conv_model)

    # Train and predict
    classifier.fit(x_train, y_train, steps=100)

    #N = y_test.shape[0]

    print("====================================================")

    print("---after fitting---")
    #y_predicted = [p['class'] for p in classifier.predict(x_test, as_iterable=True)]
    y_predicted = [p['prob'] for p in classifier.predict(x_test, as_iterable=True)]
    print("---y_predicted---")
    print(y_predicted)
    count0 = 0
    count1 = 0
    for i in y_predicted:
        if i == 1:
            count1 += 1
        else:
            count0 += 1
    print("count0", count0)
    print("count1", count1)

    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))

    s = "\nModel Report " + generate_unique_filename() +\
        "\nAccuracy : %.4g" % score +\
        "\nF1 score: %f" % metrics.f1_score(y_test, y_predicted)
        #"\nAUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob) +\


    #Print model report:
    print(s)

    filename = './output'+generate_unique_filename()+'.txt'
    file = open(filename, 'a+')

    file.write(s)


if __name__ == '__main__':
    tf.app.run()
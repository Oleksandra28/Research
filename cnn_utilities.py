__author__ = 'osopova'

import skflow
import tensorflow as tf
#from skflow.ops import *
from tensorflow.contrib import learn

F = 3
S = 1
P = 1
MAX_DOCUMENT_LENGTH = 3640
IMAGE_WIDTH = MAX_DOCUMENT_LENGTH
IMAGE_HEIGHT = 1
N_FILTERS = 20
FILTER_SHAPE1 = [3,1]
BATCH_SIZE = 257

def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 1, 1], strides=[1, S, S, 1],
        padding='SAME')

def conv_model(X, y):
    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and height
    # final dimension being the number of color channels
    print X
    print (X.get_shape())
    X = tf.reshape(X, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    print (X.get_shape())

    # first conv layer will compute N_FILTERS features for each FxF patch
    with tf.variable_scope('conv_layer1'):
        # def convolution2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', rate=1,
        h_conv1 = tf.contrib.layers.conv2d(inputs=X,
                                           num_outputs=N_FILTERS,
                                           kernel_size=FILTER_SHAPE1,
                                           padding='VALID')
        h_pool1 = max_pool_2x2(h_conv1)
        print ("h_conv1.get_shape():", h_conv1.get_shape())
        print ("h_pool1.get_shape():", h_pool1.get_shape())

    # second conv layer will compute N_FILTERS features for each FxF patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = tf.contrib.layers.conv2d(inputs=h_pool1,
                                           num_outputs=N_FILTERS,
                                           kernel_size=FILTER_SHAPE1,
                                           padding='SAME')
        h_pool2 = max_pool_2x2(h_conv2)
        print ("h_conv2.get_shape():", h_conv2.get_shape())
        print ("h_pool2.get_shape():", h_pool2.get_shape())

        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, F*F*N_FILTERS])
        print ("h_pool2_flat.get_shape():", h_pool2_flat.get_shape())

    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [IMAGE_WIDTH*IMAGE_HEIGHT], activation=tf.nn.relu, dropout=0.5)
    return skflow.models.logistic_regression(h_fc1, y)
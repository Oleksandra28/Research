__author__ = 'osopova'

import tensorflow as tf
import skflow
import numpy as np
from cnn_utilities import *
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, f_height, f_width):
    return tf.nn.conv2d(x, W, strides=[1, f_height, f_width, 1], padding='SAME')


def max_pool_2x2(x, p_height, p_width):
    return tf.nn.max_pool(x, ksize=[1, p_height, p_width, 1],
                          strides=[1, p_height, p_width, 1], padding='SAME')


data_folder = './data'

sandyData = np.loadtxt(data_folder + '/sandyData.csv', delimiter=',')
sandyLabels = np.loadtxt(data_folder + '/sandyLabels.csv', delimiter=',')

X_train, X_test, y_train, y_test = \
    train_test_split(sandyData, sandyLabels, test_size=0.2, random_state=7)

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape


### building the model
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[X_train.shape[0], X_train.shape[1]])
y_ = tf.placeholder(tf.float32, shape=[y_test.shape[0], 1])


f_height = 1
f_width = 3

p_height = 1
p_width = 2

depth1 = 32

### first conv layer

W_conv1 = weight_variable([f_height, f_width, 1, depth1])
b_conv1 = bias_variable([depth1])

# reshape x to a 4d tensor,
# with the second and third dimensions corresponding to image width and height,
# and the final dimension corresponding to the number of color channels.


h_conv1 = tf.nn.relu(conv2d(X_train, W_conv1, f_height, f_width) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1, p_height, p_width)

### second conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

### fully connected layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


### dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### read out

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

### train and eval
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = X_train[i+50*i:i+50*(i+1)]
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))


#
# # Training and predicting
#
# classifier = skflow.TensorFlowEstimator(
#     model_fn=conv_model, n_classes=2, batch_size=257, steps=20000,
#     learning_rate=0.001)
#
# classifier.fit(X_train, y_train)
# score = accuracy_score(y_test, classifier.predict(X_test))
# print("Accuracy: %f" % score)
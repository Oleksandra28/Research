__author__ = 'osopova'

import tensorflow as tf
import skflow
import numpy as np
from cnn_utilities import *
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import *


data_folder = './data'

sandyData = np.loadtxt(data_folder+'/sandyData.csv', delimiter=',')
sandyLabels = np.loadtxt(data_folder+'/sandyLabels.csv', delimiter=',')

X_train, X_test, y_train, y_test = \
    train_test_split(sandyData, sandyLabels, test_size=0.2, random_state=7)

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

# Training and predicting

classifier = skflow.TensorFlowEstimator(
    model_fn=conv_model, n_classes=2, batch_size=257, steps=20000,
    learning_rate=0.001)

classifier.fit(X_train, y_train)
score = accuracy_score(y_test, classifier.predict(X_test))
print("Accuracy: %f" % score)
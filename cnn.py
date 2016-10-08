import skflow
from sklearn import datasets, metrics
import numpy as np
from utilities import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
import tensorflow as tf

from cnn_utilities import *

### 1. split each tweet into characters
filepath0 = './CrisisLexT6/2012_Sandy_Hurricane/2012_Sandy_Hurricane_0_final.txt'
filepath1 = './CrisisLexT6/2012_Sandy_Hurricane/2012_Sandy_Hurricane_1_final.txt'

content0 = read_txt(filepath0)
content1 = read_txt(filepath1)

matrix_0 = create_matrix(content0)
matrix_1 = create_matrix(content1)

print matrix_0.shape
print matrix_1.shape

matrix_0_y = matrix_0.shape[0]*[0]
matrix_0_y = np.array(matrix_0_y)

matrix_1_y = matrix_1.shape[0]*[1]
matrix_1_y = np.array(matrix_1_y)

result_matrix = np.concatenate((matrix_0, matrix_1))
result_target = np.concatenate((matrix_0_y, matrix_1_y))

# one-hot-encode all characters
chars_set = get_character_set(result_matrix)

# dummy_vector = [[char] for char in chars_set]
#
# dummy_matrix = dummy_vector*140

dummy_vector = np.array(list(chars_set))

#dummy_vector.shape = (len(list(chars_set)),1)
#print dummy_vector.shape

# convert char to int to fit OneHotEncoder
dummy_vector = [ord(x) for x in dummy_vector]

dummy_matrix = [dummy_vector for i in range(140)]

dummy_matrix = np.array(dummy_matrix)
dummy_matrix = dummy_matrix.transpose()

encoder = OneHotEncoder(categorical_features='all', \
                        sparse=False, \
                        handle_unknown='ignore')

encoder.fit(dummy_matrix)

result_matrix = [[ord(x) if x != '' else 0 for x in row] for row in result_matrix]


result_matrix = encoder.transform(result_matrix)

print result_matrix.shape # (9013, 9240)

X_train, X_test, y_train, y_test = train_test_split(result_matrix, result_target, test_size=0.33, random_state=42)

#-------------------------------------------------------------------

# Training and predicting
classifier = skflow.TensorFlowEstimator(
    model_fn=conv_model, n_classes=2, batch_size=88, steps=20000,
    learning_rate=0.001)

classifier.fit(X_train, y_train)
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print("Accuracy: %f" % score)
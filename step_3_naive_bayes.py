__author__ = 'osopova'

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.cross_validation import train_test_split, cross_val_score
from utilities import *

data_folder = './data'

sandyData = np.loadtxt(data_folder+'/sandyData.csv', delimiter=',')
sandyLabels = np.loadtxt(data_folder+'/sandyLabels.csv', delimiter=',')
#
# albertaData = np.loadtxt(data_folder+'/albertaData.csv', delimiter=',')
# albertaLabels = np.loadtxt(data_folder+'/albertaLabels.csv', delimiter=',')
#
# bostonData = np.loadtxt(data_folder+'/bostonData.csv', delimiter=',')
# bostonLabels = np.loadtxt(data_folder+'/bostonLabels.csv', delimiter=',')
#
# oklahomaData = np.loadtxt(data_folder+'/oklahomaData.csv', delimiter=',')
# oklahomaLabels = np.loadtxt(data_folder+'/oklahomaLabels.csv', delimiter=',')
#
# queenslandData = np.loadtxt(data_folder+'/queenslandData.csv', delimiter=',')
# queenslandLabels = np.loadtxt(data_folder+'/queenslandLabels.csv', delimiter=',')
#
# westtexasData = np.loadtxt(data_folder+'/westtexasData.csv', delimiter=',')
# westtexasLabels = np.loadtxt(data_folder+'/westtexasLabels.csv', delimiter=',')

#-----------------------------------------------------------------------------------------------

nb_results_folder = './nb_results'

### Naive Bayes Classification

gnb = BernoulliNB()

### shuffle by setting test_size = 0
X_train, X_test, y_train, y_test = \
    train_test_split(sandyData, sandyLabels, test_size=0, random_state=7)
n_folds = 5
scores = cross_val_score(gnb, sandyData, sandyLabels, cv=n_folds)
print scores

generate_report(classifier='BernoulliNB', scores=scores)




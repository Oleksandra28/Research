__author__ = 'osopova'

import numpy as np
import string
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
import pickle
from os import path
import sys

#--------------------------------------------------------------------------------------
def read_txt(filepath):
    with open(filepath) as f:
        original_content = f.readlines()
    return original_content

#--------------------------------------------------------------------------------------
def clean_tweets(original_content, lower=True):
    l = len(original_content)
    content = [x.strip('\n') for x in original_content]
    assert(l == len(content))

    ### split by character
    content = [list(x) for x in content]

    if lower:
        content = [[x.lower() for x in row] for row in content]

    ### keep wanted characters
    content = keep_wanted_chars(content)

    ### convert to numpy array
    content = np.asarray(content)
    return content

#--------------------------------------------------------------------------------------
def create_matrix(content, tweet_length=140):
    n_samples = len(content)
    result_matrix = []
    for tweet in content:
        print "tweet", tweet
        l = len(tweet)
        if (l > tweet_length):
            tweet = tweet[:tweet_length]
        l = len(tweet)
        assert(len(tweet) <= tweet_length)
        zeros_needed = tweet_length-l
        values = ['']*zeros_needed
        tweet = np.append(tweet, values)
        assert(len(tweet) == tweet_length)
        print "new tweet", tweet
        result_matrix.append(tweet)

    result_matrix = np.array(result_matrix)
    assert (result_matrix.shape == (n_samples, tweet_length))
    return result_matrix

    return result

#--------------------------------------------------------------------------------------
def wanted_chars(lower=True):

    if lower:
        letters = string.ascii_lowercase
    else:
        letters = string.ascii_lowercase + string.ascii_uppercase

    result = [c for c in letters+" "]
    return result

#--------------------------------------------------------------------------------------
def wanted_chars_numeric(upper=False):

    letters = wanted_chars()
    letters = [ord(x) for x in letters]
    return letters

#--------------------------------------------------------------------------------------
def keep_wanted_chars(content, lower=True):
    wanted = wanted_chars(lower)
    new_content = []
    for row in content:
        new_row = [c for c in row if c in wanted]
        new_content.append(new_row)
    return new_content

#--------------------------------------------------------------------------------------
def load_clean_dataset(filepath0, filepath1, lower=True, tweet_length=140):

    content0 = read_txt(filepath0)
    content1 = read_txt(filepath1)

    content0 = clean_tweets(content0, lower)
    content1 = clean_tweets(content1, lower)

    final_min_0, final_max_0, mode_0, mean_0 = get_max_tweet_length(content0)
    final_min_1, final_max_1, mode_1, mean_1 = get_max_tweet_length(content1)

    print final_min_0, final_max_0, mode_0, mean_0
    print final_min_1, final_max_1, mode_1, mean_1

    tweet_length = 70
    print "tweet_length", tweet_length

    matrix_0 = create_matrix(content0, tweet_length)
    matrix_1 = create_matrix(content1, tweet_length)

    matrix_0_y = matrix_0.shape[0]*[0]
    matrix_0_y = np.array(matrix_0_y)

    matrix_1_y = matrix_1.shape[0]*[1]
    matrix_1_y = np.array(matrix_1_y)

    assert(matrix_0.shape[0] == matrix_0_y.shape[0])
    assert(matrix_1.shape[0] == matrix_1_y.shape[0])

    result_matrix = np.concatenate((matrix_0, matrix_1))
    result_target = np.concatenate((matrix_0_y, matrix_1_y))

    return result_matrix, result_target

#--------------------------------------------------------------------------------------
def get_encoder(lower=True, n_columns=140):

    encoder_filename = "./encoder-pickle.dat"
    encoder_exists = path.isfile(encoder_filename)

    if encoder_exists:
        ### pickle encoder
        encoder = pickle.load(open(encoder_filename, "rb"))
    else:
        encoder = OneHotEncoder(categorical_features='all', \
                            sparse=False, \
                            handle_unknown='ignore')
        pickle.dump(encoder, open(encoder_filename, "wb"))

    ### encode as One-Hot
    dummy_vector = np.array(wanted_chars_numeric(upper=not lower))
    dummy_matrix = [dummy_vector for i in range(n_columns)]
    dummy_matrix = np.array(dummy_matrix)
    dummy_matrix = dummy_matrix.transpose()
    encoder.fit(dummy_matrix)

    return encoder

#--------------------------------------------------------------------------------------
def encode_one_hot(matrix, lower=True, chars_to_ascii=True):

    if chars_to_ascii:
        matrix = [[ord(x) if x != '' else 0 for x in row] for row in matrix]

    n_cols = len(matrix[0])
    print "n_cols", n_cols
    encoder = get_encoder(lower=lower, n_columns=n_cols)
    result_matrix = encoder.transform(matrix)
    return result_matrix

#--------------------------------------------------------------------------------------
def get_clean_data_clean_labels(filepath0, filepath1):

    data, labels = load_clean_dataset(filepath0, filepath1)

    sizeData = data.shape
    sizeLabels = labels.shape
    print "labels", labels.shape
    print labels

    data = encode_one_hot(data)

    # one-hot encode labels
    encoder_filename = "./binary-labels-encoder-pickle.dat"
    encoder_exists = path.isfile(encoder_filename)
    if encoder_exists:
        ### pickle encoder
        encoder = pickle.load(open(encoder_filename, "rb"))
    else:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        pickle.dump(encoder, open(encoder_filename, "wb"))

    labels = np.asarray(labels)
    labels = labels.reshape(-1,1)
    labels = encoder.fit_transform(labels)

    print encoder.n_values_
    print encoder.feature_indices_

    print "labels", labels.shape
    print labels
    #labels = labels.reshape(-1, 1)


    assert(sizeData[0] == data.shape[0])
    #assert(sizeLabels[0] == labels.shape[0])
    return data, labels

#--------------------------------------------------------------------------------------
def generate_unique_filename():
    fmt = "%Y-%m-%d-%H-%M-%S"
    filename = time.strftime(fmt)
    filename = '/'+filename
    return filename

#--------------------------------------------------------------------------------------
def generate_report(classifier, scores, PCA=False,
                    test_size=0, random_state=7,
                    n_folds=5,
                    folder='./', extension='.csv'):

    result_df = pd.DataFrame()
    result_df.loc[0, 'classifier'] = classifier
    result_df.loc[0, 'PCA'] = PCA
    result_df.loc[0, 'test_size'] = test_size
    result_df.loc[0, 'random_state'] = random_state

    result_df.loc[0, 'cv'] = n_folds
    for fold in range(0, n_folds):
        s = 'fold_'+ str(fold)
        result_df.loc[0, s] = scores[fold]
    result_df.to_csv(folder+generate_unique_filename()+extension)
    print 'report has been generated'

#--------------------------------------------------------------------------------------

def get_max_tweet_length(content):

    all_lengths = []
    for row in content:
        all_lengths.append(len(row))

    final_max = max(all_lengths)
    final_min = min(all_lengths)
    mode = max(set(all_lengths), key=all_lengths.count)
    mean = np.mean(all_lengths)

    return final_min, final_max, mode, mean






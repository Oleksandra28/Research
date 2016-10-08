__author__ = 'osopova'

import numpy as np
import string
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd

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

    ### keep wanted characters
    content = keep_wanted_chars(content, not lower)

    if lower:
        content = [[x.lower() for x in row] for row in content]

    ### convert to numpy array
    content = np.asarray(content)
    return content

#--------------------------------------------------------------------------------------
def create_matrix(content, tweet_length=140):
    n_samples = len(content)
    result_matrix = []
    for tweet in content:
        l = len(tweet)
        if (l > tweet_length):
            tweet = tweet[:tweet_length]
        l = len(tweet)
        assert(len(tweet) <= tweet_length)
        zeros_needed = tweet_length-l
        values = ['']*zeros_needed
        tweet = np.append(tweet, values)
        assert(len(tweet) == tweet_length)
        result_matrix.append(tweet)

    result_matrix = np.array(result_matrix)
    assert (result_matrix.shape == (n_samples, tweet_length))
    return result_matrix

    return result

#--------------------------------------------------------------------------------------
def wanted_chars(upper=False):

    letters = string.ascii_lowercase

    if upper:
        letters = string.ascii_lowercase + string.ascii_uppercase

    result = [c for c in letters]
    return result

#--------------------------------------------------------------------------------------
def wanted_chars_numeric(upper=False):

    letters = wanted_chars()
    letters = [ord(x) for x in letters]
    return letters

#--------------------------------------------------------------------------------------
def keep_wanted_chars(content, upper=False):
    wanted = wanted_chars(upper)
    new_content = []
    for row in content:
        new_row = [c for c in row if c in wanted]
        new_content.append(new_row)
    return new_content

#--------------------------------------------------------------------------------------
def load_clean_dataset(filepath0,filepath1, lower=True):

    content0 = read_txt(filepath0)
    content1 = read_txt(filepath1)

    content0 = clean_tweets(content0, lower)
    content1 = clean_tweets(content1, lower)

    matrix_0 = create_matrix(content0)
    matrix_1 = create_matrix(content1)

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

    encoder_exists = False

    if encoder_exists:
        ### pickle encoder
        encoder = None
    else:
        encoder = OneHotEncoder(categorical_features='all', \
                            sparse=False, \
                            handle_unknown='ignore')

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

    encoder = get_encoder(lower=lower)
    result_matrix = encoder.transform(matrix)
    return result_matrix

#--------------------------------------------------------------------------------------
def get_data_labels(filepath0, filepath1):

    data, labels = load_clean_dataset(filepath0, filepath1)

    sizeData = data.shape
    sizeLabels = labels.shape

    data = encode_one_hot(data)

    assert(sizeData[0] == data.shape[0])
    assert(sizeLabels[0] == labels.shape[0])
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










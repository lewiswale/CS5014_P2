def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def read_training_data(directory):
    X = pd.read_csv(directory + '/X.csv', header=None)
    print(X)

    y = pd.read_csv(directory + '/y.csv', header=None)
    print(y)

    to_predict = pd.read_csv(directory + '/XToClassify.csv', header=None)
    print(to_predict)

    return X, y, to_predict


def train_binary_svc(X, y):
    clf = SVC(random_state=3251)
    clf.fit(X, y)
    return clf


def train_multiclass_svc(X, y):
    clf = SVC(gamma=0.1, random_state=3251)
    clf.fit(X, y)
    return clf


def svc_predict(clf, to_predict):
    predictions = clf.predict(to_predict)
    print('Predicted:\n{}'.format(predictions))


def train_binary_nn(X, y):
    clf = MLPClassifier(random_state=3251)
    clf.fit(X, y)
    return clf


def train_multiclass_nn(X, y):
    clf = MLPClassifier(hidden_layer_sizes=150,random_state=3251)
    clf.fit(X, y)
    return clf



def nn_predict(clf, to_predict):
    predictions = clf.predict(to_predict)
    print('Predicted:\n{}'.format(predictions))


def score_classifier(clf, X, y):
    score = clf.score(X, y)
    print('Score: {}'.format(score))


def cross_val_score_classifier(clf, X, y):
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    average = scores.mean()
    print(average)


if __name__ == "__main__":
    print('============================================')
    print('Reading binary data...')
    X, y, to_predict = read_training_data('binary')
    print('\nBinary data read.')

    print('============================================')
    print('Training SVM...')
    bin_svc_clf = train_binary_svc(X, y)
    print('Scoring SVM...')
    score_classifier(bin_svc_clf, X, y)
    cross_val_score_classifier(bin_svc_clf, X, y)
    print('Making predictions...')
    svc_predict(bin_svc_clf, to_predict)

    print('============================================')
    print('Training NN...')
    bin_nn_clf = train_binary_nn(X, y)
    print('Scoring NN...')
    score_classifier(bin_nn_clf, X, y)
    cross_val_score_classifier(bin_nn_clf, X, y)
    print('Making predictions...')
    nn_predict(bin_nn_clf, to_predict)

    print('============================================')
    print('Reading Multiclass data...')
    X, y, to_predict = read_training_data('multiclass')
    print('\nMulticlass data read.')

    print('============================================')
    print('Training SVM...')
    mul_svc_clf = train_multiclass_svc(X, y)
    print('Scoring SVM...')
    score_classifier(mul_svc_clf, X, y)
    cross_val_score_classifier(mul_svc_clf, X, y)
    print('Making predictions...')
    svc_predict(mul_svc_clf, to_predict)

    print('============================================')
    print('Training NN...')
    mul_nn_clf = train_multiclass_nn(X, y)
    print('Scoring NN...')
    score_classifier(mul_nn_clf, X, y)
    cross_val_score_classifier(mul_nn_clf, X, y)
    print('Making predictions...')
    svc_predict(mul_nn_clf, to_predict)







def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
    gammas = [10, 5, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    c_vals = [100, 50, 20, 10, 5, 2, 1, 0.5, 0.1]
    scores = []
    gammas_to_plot = []
    c_to_plot = []
    best_score = 0
    best_clf = None

    for gamma in gammas:
        for c in c_vals:
            current_clf = SVC(gamma=gamma, C=c, random_state=3251)
            current_score = cross_val_score_classifier(current_clf, X, y)
            scores.append(current_score)
            gammas_to_plot.append(gamma)
            c_to_plot.append(c)
            if current_score > best_score:
                best_score = current_score
                best_clf = current_clf

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(gammas_to_plot, c_to_plot, scores, 'blue')
    ax.set_xlabel('Gamma Value')
    ax.set_ylabel('C Value')
    ax.set_zlabel('Accuracy')
    ax.set_title('How Gamma and C values affect SVM accuracy')
    plt.savefig('svm_testing')

    best_clf.fit(X, y)
    return best_clf


def svc_predict(clf, to_predict):
    predictions = clf.predict(to_predict)
    print('SVC Predicted:\n{}'.format(predictions))
    return predictions


def train_binary_nn(X, y):
    clf = MLPClassifier(random_state=3251)
    clf.fit(X, y)
    return clf


def train_multiclass_nn(X, y):
    layer_sizes = [1, 2, 5, 10, 20, 50, 100, 150, 200, 500, 1000]
    learning_inits = [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
    scores = []
    sizes_to_plot = []
    inits_to_plot = []
    best_score = 0
    best_clf = None

    for layer_size in layer_sizes:
        for init in learning_inits:
            current_clf = MLPClassifier(hidden_layer_sizes=layer_size, learning_rate_init=init, random_state=3251)
            current_score = cross_val_score_classifier(current_clf, X, y)
            scores.append(current_score)
            sizes_to_plot.append(layer_size)
            inits_to_plot.append(init)

            if current_score > best_score:
                best_score = current_score
                best_clf = current_clf

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(sizes_to_plot, inits_to_plot, scores, 'blue')
    ax.set_xlabel('Hidden Layer size')
    ax.set_ylabel('Learning rate initial value')
    ax.set_zlabel('Accuracy')
    ax.set_title('How Hidden Layer Size and Initial Learning Rate values affect NN accuracy')
    plt.savefig('nn_testing')

    best_clf.fit(X, y)
    return best_clf



def nn_predict(clf, to_predict):
    predictions = clf.predict(to_predict)
    print('NN Predicted:\n{}'.format(predictions))
    return predictions


def score_classifier(clf, X, y):
    score = clf.score(X, y)
    print('Score: {}'.format(score))
    return score


def cross_val_score_classifier(clf, X, y):
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    average = scores.mean()
    print(average)
    return average


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
    svc_predict(bin_svc_clf, to_predict)

    print('============================================')
    print('Training NN...')
    bin_nn_clf = train_binary_nn(X, y)
    print('Scoring NN...')
    score_classifier(bin_nn_clf, X, y)
    cross_val_score_classifier(bin_nn_clf, X, y)

    print('============================================')
    print('Making predictions...')
    predicted_classes = [nn_predict(bin_nn_clf, to_predict), svc_predict(bin_svc_clf, to_predict)]
    df = pd.DataFrame(predicted_classes)
    file_name = 'binary/PredictedClasses.csv'
    df.to_csv(file_name, index=False)

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


    print('============================================')
    print('Training NN...')
    mul_nn_clf = train_multiclass_nn(X, y)
    print('Scoring NN...')
    score_classifier(mul_nn_clf, X, y)
    cross_val_score_classifier(mul_nn_clf, X, y)

    print('============================================')
    print('Making predictions...')
    predicted_classes = [nn_predict(mul_nn_clf, to_predict), svc_predict(mul_svc_clf, to_predict)]
    df = pd.DataFrame(predicted_classes)
    file_name = 'multiclass/PredictedClasses.csv'
    df.to_csv(file_name, index=False)








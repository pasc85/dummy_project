from random import randint as ri
import numpy as np
from sklearn.neural_network import MLPClassifier

global n_bits
n_bits = 21


def conv(i):
    fs = '{0:0' + str(n_bits) + 'b}' 
    t = fs.format(i)
    return [int(c) for c in t]


def is_multiple_of_3(i):
    if i % 3 == 0:
        return True
    else:
        return False


def generate_data(size):
    x = []
    y = []
    for i in range(size):
        t = ri(1, 2**n_bits)
        x.append(t)
        y.append(int(is_multiple_of_3(t)))
    X = [conv(t) for t in x]
    X = np.asarray(X)
    y = np.ravel(y)
    return X, y


def accuracy_hls_config(X_train, y_train, X_test, y_test, hls=(n_bits,)):
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hls)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = sum(y_test == y_pred)/len(y_test)
    return accuracy

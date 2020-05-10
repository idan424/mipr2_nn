import numpy as np
import os
import matplotlib.pyplot as plt


def data_load(path):
    trn_path = path + '/ms_classification_project/training/'
    val_path = path + '/ms_classification_project/validation/'

    x_trn, y_trn = load_dir(trn_path)
    x_trn, y_trn = randomize(x_trn, y_trn)

    x_val, y_val = load_dir(val_path)

    return x_trn, y_trn, x_val, y_val


def randomize(x, y):
    rnd = np.arange(x.shape[1])
    np.random.shuffle(rnd)
    return x[:, rnd], y[rnd]


def load_dir(path):
    name_list = os.listdir(path)
    X, Y = [], []
    for name in name_list:
        x = plt.imread(path + name)
        y = 0
        if 'pos' in name: y = 1
        X.append(x.flatten())
        Y.append(y)
    return np.array(X).T, np.array(Y)


def relu(x):
    if x <= 0: return 0
    return x


def drelu(x):
    if x <= 0: return 0
    return 1


def mod_relu(x):
    if x <= 0: return 0.1 * x
    return x


def dmod_relu(x):
    if x <= 0: return 0.1
    return 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 - np.tanh(x) ** 2


def thresh(x):
    if x >= 0.5: return 1
    return 0

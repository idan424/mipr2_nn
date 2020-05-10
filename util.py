import numpy as np
import os
import matplotlib.pyplot as plt

mean = lambda x: sum(x) / len(x)
floss, dfloss = lambda y, yp: (y - yp) ** 2, lambda y, yp: (y - yp) * 2


def data_load(path):
    x_trn, y_trn = load_dir(path + '/ms_classification_project/training/')
    x_val, y_val = load_dir(path + '/ms_classification_project/validation/')

    return x_trn, y_trn, x_val, y_val


def randomize(x, y):
    rnd = np.arange(x.shape[1])
    np.random.shuffle(rnd)
    return x[:, rnd], y[rnd]


def load_dir(path):
    picture_list = os.listdir(path)
    X, Y = np.ndarray([1024, 0]), []
    for pic in picture_list:
        X = np.c_[X, plt.imread(path + pic).flatten().reshape([1024, 1])]
        Y.append(int('pos' in pic))
    return X, np.array(Y)


def relu(x):
    if x <= 0: return 0
    return x


def drelu(x):
    if x <= 0: return 0
    return 1


def leaky_relu(x):
    if x <= 0: return 0.1 * x
    return x


def dleaky_relu(x):
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

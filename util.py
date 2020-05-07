import numpy as np
import os
import matplotlib.pyplot as plt


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


def data_load(path):
    trn_path = path + '/ms_classification_project/training/'
    train_name_list = os.listdir(trn_path)
    Xt = []
    Yt = []
    for name in train_name_list:
        x = plt.imread(trn_path + name)
        y = 0
        if 'pos' in name: y = 1
        Xt.append(x.flatten())
        Yt.append(y)
    x_trn = np.array(Xt).T
    y_trn = np.array(Yt)

    randt = np.arange(x_trn.shape[1])
    np.random.shuffle(randt)
    x_trn = x_trn[:, randt]
    y_trn = y_trn[randt]

    val_path = path + '/ms_classification_project/validation/'
    val_name_list = os.listdir(val_path)
    Xv = []
    Yv = []
    for name in val_name_list:
        x = plt.imread(val_path + name)
        y = 0
        if name.find('pos'): y = 1
        Xv.append(x.flatten())
        Yv.append(y)
    x_val = np.array(Xv).T
    y_val = np.array(Yv)

    randv = np.arange(x_val.shape[1])
    np.random.shuffle(randv)
    x_val = x_val[:, randv]
    y_val = y_val[randv]

    return x_trn, y_trn, x_val, y_val
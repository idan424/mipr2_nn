import numpy as np
import matplotlib.pyplot as plt
import os

path = 'C:\\Users\\עידן\\Dropbox\\BioBioMoach\\שנה ג\\סמסטר ב\\ה-תמונות 2\\שיעורי בית'
sub = '\\mipr_project\\ms_classification_project\\training\\'

N_hidden = 10
mini_batch_size = 32
STEP_SIZE = 1e-7


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


class CancerClassifier:
    def __init__(self, x_trn, y_trn, x_val, y_val):
        self.trn_data = x_trn, y_trn
        self.val_data = x_val, y_val

        self.w1 = np.random.randn(N_hidden, 1024)
        self.b1 = np.random.randn(N_hidden, mini_batch_size)

        self.w2 = np.random.randn(1, N_hidden)
        self.b2 = np.random.randn(1, mini_batch_size)

    def mini_batch_step(self, x=None, y=None, train=False):
        """
            x of size 1024*mini_batch_size
            y of size 1*mini_batch_size
        """
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.vectorize(mod_relu)(z1)

        z2 = np.dot(self.w2, a1) + self.b2
        a2 = np.vectorize(sigmoid)(z2)

        cost = (a2 - y) ** 2

        if train:
            acc = self.backpropagate(cost, a2, y, z2, a1, z1, x)
            print(f'cost: {np.mean(np.sqrt(cost))}, accuracy:{acc}')

    def backpropagate(self, cost, a2, y, z2, a1, z1, x):
        # dC/da2
        dc_da2 = 2 * (a2 - y)
        # da2/dz2
        da2_dz2 = np.vectorize(dsigmoid)(z2)
        # dz2/dw2 = a1

        # dz2/da1
        dz2_da1 = self.w2.T
        # da1/dz1
        da1_dz1 = np.vectorize(dmod_relu)(z1)
        # dz1/dw1 = x

        # dC/dw2 = dC/da2 * da2/dz2 * dz2/dw2
        dw2 = np.dot(dc_da2 * da2_dz2, a1.T)
        # dC/db2 = dC/da2 * da2/dz2 * dz2/db2
        db2 = dc_da2 * da2_dz2 * 1

        # dC/dw1 = dC/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
        dw1 = np.dot(np.dot(dz2_da1, dc_da2 * da2_dz2) * da1_dz1, x.T)
        # dC/dw1 = dC/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/db1
        db1 = dc_da2 * da2_dz2 * dz2_da1 * da1_dz1

        self.update(dw1, db1, dw2, db2)

        acc = 1 - np.sum(abs(y - np.vectorize(round)(a2))) / mini_batch_size
        return acc

    def update(self, dw1, db1, dw2, db2):
        self.w1 = self.w1 - dw1 * STEP_SIZE
        self.w2 = self.w2 - dw2 * STEP_SIZE
        self.b1 = self.b1 - db1 * STEP_SIZE
        self.b2 = self.b2 - db2 * STEP_SIZE

    def go_over_data(self, x_trn, y_trn):
        num_batch = int(len(y_trn) / mini_batch_size)
        rem = len(y_trn) % mini_batch_size
        for i in range(num_batch):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            # print(f'batch number{i}')
            self.mini_batch_step(x_trn[:, start:end], y_trn[start:end], train=True)
        if rem > 0: self.mini_batch_step(x_trn[:, end:end + rem], y_trn[end:end + rem])
        pass

    def train_for(self, n_epoch):
        for i in range(n_epoch):
            print('##########################################')
            print(f'epoch # {i + 1}')
            print('##########################################')
            self.go_over_data(*self.trn_data)

    # TODO: make it work
    def validate(self):
        x, y = self.val_data
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.vectorize(mod_relu)(z1)

        z2 = np.dot(self.w2, a1) + self.b2
        a2 = np.vectorize(sigmoid)(z2)

        cost = (a2 - y)


def data_load():
    trn_path = path + '/mipr_project/ms_classification_project/training/'
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

    val_path = path + '/mipr_project/ms_classification_project/validation/'
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

    return x_trn, y_trn, x_val, y_val


if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid = data_load()

    cc = CancerClassifier(*data_load())
    # cc.mini_batch_step(x_train[:, 256 - int(mini_batch_size / 2):256 + int(mini_batch_size / 2)],
    #                    y_train[256 - int(mini_batch_size / 2):256 + int(mini_batch_size / 2)])
    cc.train_for(10)
    # cc.validate()
    self = cc

from util import *

# np.random.seed(11)  # seed for reproducability - will be deprecated

N_hidden = 4
mini_batch_size = 32
STEP_SIZE = 1e-2


class CancerClassifier:
    def __init__(self, x_trn, y_trn, x_val, y_val):
        self.STEP_SIZE = STEP_SIZE

        self.trn_data = x_trn, y_trn
        self.val_data = x_val, y_val

        self.w1 = np.random.randn(N_hidden, 1024)
        self.b1 = np.random.randn(N_hidden, 1)

        self.w2 = np.random.randn(1, N_hidden)
        self.b2 = np.random.randn(1, 1)

    def mini_batch_step(self, x=None, y=None, train=False):
        """
            x shape =  [1024, mini_batch_size]
            y shape =  [1, mini_batch_size]
        """
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.vectorize(mod_relu)(z1)

        z2 = np.dot(self.w2, a1) + self.b2
        a2 = np.vectorize(sigmoid)(z2)

        diff = a2 - y
        cost = diff ** 2
        loss = np.sqrt(cost)
        a2rnd = np.vectorize(round)(a2)
        acc_diff = abs(y - a2rnd)
        acc = 1 - (np.sum(acc_diff) / mini_batch_size)

        if train:
            self.backpropagate(x, y, z1, a1, z2, a2, cost)
            # print(f'mini_bach_cost: {np.mean(loss)}, mini_bach_accuracy:{acc}')

        return np.mean(loss), acc

    def backpropagate(self, x, y, z1, a1, z2, a2, cost):
        # dC/da2
        dc_da2 = 2 * (a2 - y)

        # da2/dz2
        da2_dz2 = np.vectorize(dsigmoid)(z2)

        # dz2/dw2
        dz2_dw2 = a1.T
        # dz2/db2 = 1

        # dz2/da1
        dz2_da1 = self.w2.T

        # da1/dz1
        da1_dz1 = np.vectorize(dmod_relu)(z1)

        # dz1/dw1
        dz1_dw1 = x.T  # dz1/db1 = 1




        # dC/dw2 = dC/da2 * da2/dz2 * dz2/dw2
        dw2 = np.dot(dc_da2 * da2_dz2, dz2_dw2)

        # dC/db2 = dC/da2 * da2/dz2 * dz2/db2
        db2 = np.mean(dc_da2 * da2_dz2, axis=1).reshape(1, 1)

        # dC/dw1 = dC/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
        dw1 = np.dot(np.dot(dz2_da1, dc_da2 * da2_dz2) * da1_dz1, dz1_dw1)

        # dC/db1 = dC/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/db1
        db1 = np.mean(dc_da2 * da2_dz2 * dz2_da1 * da1_dz1, axis=1).reshape(N_hidden, 1)

        self.update(dw1, db1, dw2, db2)

    def update(self, dw1, db1, dw2, db2):
        self.w1 = self.w1 - dw1 * self.STEP_SIZE
        self.w2 = self.w2 - dw2 * self.STEP_SIZE
        self.b1 = self.b1 - db1 * self.STEP_SIZE
        self.b2 = self.b2 - db2 * self.STEP_SIZE

    def epoch(self, x_trn, y_trn):
        num_batch = int(len(y_trn) / mini_batch_size)
        loss, acc = [], []
        for i in range(num_batch):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            l_mb, acc_mb = self.mini_batch_step(x_trn[:, start:end], y_trn[start:end], train=True)
            loss.append(l_mb), acc.append(acc_mb)

        # print(f'total loss:{np.mean(loss)}, total accuracy:{np.mean(acc)}')

        if np.mean(loss) < 0.1: self.STEP_SIZE = 1e-3

        return np.mean(loss), np.mean(acc)

    def train_for(self, n_epoch):
        loss, acc = [], []
        for i in range(n_epoch):
            # print(f'epoch #{i + 1}')
            l_ep, a_ep = self.epoch(*self.trn_data)
            loss.append(l_ep), acc.append(a_ep)
        print(f'{n_epoch} epochs loss: {np.mean(loss)}, {n_epoch} epochs accuracy: {np.mean(acc)}')
        return np.mean(loss), np.mean(acc)

    def validate(self):
        loss, acc = [], []
        xs, ys = self.val_data
        for x, y in zip(xs.T, ys):
            l, a = self.val_sample(x, y)
            loss.append(l), acc.append(a)
        print(f'total val loss: {np.mean(loss)}, total val accuracy: {np.mean(acc)}')
        return np.mean(loss), np.mean(acc)

    def val_sample(self, x, y):
        z1 = np.dot(self.w1, x) + np.mean(self.b1)
        a1 = np.vectorize(mod_relu)(z1)

        z2 = np.dot(self.w2, a1) + np.mean(self.b2)
        a2 = np.vectorize(sigmoid)(z2)

        cost = (a2 - y) ** 2
        loss = np.sqrt(cost)

        acc = 1 - abs(y - np.vectorize(round)(a2))
        return loss[0], acc[0]


if __name__ == "__main__":
    cc = CancerClassifier(*data_load(os.getcwd()))

    tot_ep = 500
    ep_per_run = 25

    trn_loss, trn_acc, val_loss, val_acc = [], [], [], []
    for _ in range(int(tot_ep / ep_per_run)):
        tl, ta = cc.train_for(ep_per_run)
        vl, va = cc.validate()
        trn_loss.append(tl), trn_acc.append(ta), val_loss.append(vl), val_acc.append(va)

    plt.figure(), plt.plot(trn_loss, label='train'), plt.plot(val_loss, label='validation'), plt.legend(), plt.show()
    self = cc

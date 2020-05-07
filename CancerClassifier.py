from util import *

# np.random.seed(11)  # seed for reproducability - will be deprecated

N_hidden = 4  # bigger is slower
mini_batch_size = 32  # bigger is faster
STEP_SIZE = 1e-3

tot_ep = 1000  # total number of epochs
ep_per_run = 20  # epochs between loss check


class CancerClassifier:
    def __init__(self, x_trn, y_trn, x_val, y_val):
        self.STEP_SIZE = STEP_SIZE

        self.trn_data = x_trn, y_trn
        self.val_data = x_val, y_val

        self.w1 = np.random.randn(N_hidden, 1024)
        self.b1 = np.random.randn(N_hidden, 1)

        self.w2 = np.random.randn(1, N_hidden)
        self.b2 = np.random.randn(1, 1)

    def train_batch(self, x, y):
        """
            x shape =  [1024, mini_batch_size]
            y shape =  [1, mini_batch_size]
        """
        # layer 1 - input
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.vectorize(mod_relu)(z1)

        # layer 2 - hidden
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = np.vectorize(sigmoid)(z2)

        # layer 3 - output
        y_pred = np.vectorize(round)(a2)

        loss = np.sum((a2 - y) ** 2) / mini_batch_size
        acc_diff = abs(y - y_pred)
        acc = 1 - (np.sum(acc_diff) / mini_batch_size)

        self.backpropagate(x, y, z1, a1, z2, a2)

        return loss, acc

    def backpropagate(self, x, y, z1, a1, z2, a2):
        # general gradients
        dc_da2 = 2 * (a2 - y)  # dC/da2
        da2_dz2 = np.vectorize(dsigmoid)(z2)  # da2/dz2
        dz2_da1 = self.w2.T  # dz2/da1
        da1_dz1 = np.vectorize(dmod_relu)(z1)  # da1/dz1

        # layer gradients
        dz2_dw2 = a1.T  # dz2/dw2
        dz2_db2 = np.ones([mini_batch_size, 1])  # dz2/db2
        dz1_dw1 = x.T  # dz1/dw1
        dz1_db1 = np.ones([mini_batch_size, 1])  # dz1/db1

        # layer 2 gradients
        dw2 = np.dot(dc_da2 * da2_dz2, dz2_dw2)  # dC/dw2 = dC/da2 * da2/dz2 * dz2/dw2
        db2 = np.dot(dc_da2 * da2_dz2, dz2_db2)  # dC/db2 = dC/da2 * da2/dz2 * dz2/db2

        # layer to layer gradient
        dc_da1 = np.dot(dz2_da1, dc_da2 * da2_dz2)

        # layer 1 gradients
        dw1 = np.dot(dc_da1 * da1_dz1, dz1_dw1)  # dC/dw1 = dC/da1 * da1/dz1 * dz1/dw1
        db1 = np.dot(dc_da1 * da1_dz1, dz1_db1)  # dC/db1 = dC/da1 * da1/dz1 * dz1/db1

        # actual descent
        self.update(dw1, db1, dw2, db2)

    def update(self, dw1, db1, dw2, db2):
        for d in [dw1, db1, dw2, db2]:
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    if d[i, j] < 1e-9: d[i, j] = 0
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
            l, a = self.train_batch(x_trn[:, start:end], y_trn[start:end])
            loss.append(l), acc.append(a)

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
            l, a = self.val_sample(x.reshape([len(x), 1]), y)
            loss.append(l), acc.append(a)
        print(f'total val loss: {np.mean(loss)}, total val accuracy: {np.mean(acc)}\n')
        return np.mean(loss), np.mean(acc)

    def val_sample(self, x, y):
        z1 = np.dot(self.w1, x)  # + self.b1
        a1 = np.vectorize(mod_relu)(z1)

        z2 = np.dot(self.w2, a1) + self.b2
        a2 = np.vectorize(sigmoid)(z2)

        cost = (a2 - y) ** 2
        loss = np.sqrt(cost)

        acc = 1 - abs(y - np.vectorize(round)(a2))
        return loss[0], acc[0]


if __name__ == "__main__":
    cc = CancerClassifier(*data_load(os.getcwd()))

    trn_loss, trn_acc, val_loss, val_acc = [], [], [], []
    for _ in range(int(tot_ep / ep_per_run)):
        tl, ta = cc.train_for(ep_per_run)
        vl, va = cc.validate()
        trn_loss.append(tl), trn_acc.append(ta), val_loss.append(vl), val_acc.append(va)

    plt.figure(), plt.plot(trn_loss, label='train'), plt.plot(val_loss, label='validation')
    plt.title("loss"), plt.legend(), plt.show()
    self = cc

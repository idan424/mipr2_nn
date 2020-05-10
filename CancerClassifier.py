from util import *

# np.random.seed(1)  # seed for reproducability - will be deprecated

N_hidden = 16  # bigger is slower
mini_batch_size = 32  # bigger is faster
STEP_SIZE = 1e-3
min_th = 1e-9

tot_ep = 400  # total number of epochs

f1 = np.vectorize(mod_relu)  # acts on z1
df1 = np.vectorize(dmod_relu)

f2 = np.vectorize(sigmoid)  # acts on z2
df2 = np.vectorize(dsigmoid)

f3 = np.vectorize(sigmoid)  # acts on a2

mean = lambda x: sum(x) / len(x)
vround = np.vectorize(round)

floss = lambda y, yp: (yp - y) ** 2
dfloss = lambda y, yp: (yp - y) * 2


class CancerClassifier(object):
    def __init__(self, x_trn, y_trn, x_val, y_val):
        self.STEP_SIZE = STEP_SIZE

        self.trn_data = x_trn, y_trn
        self.val_data = x_val, y_val

        self.w1 = np.random.randn(N_hidden, 1024)  # / 1024
        self.b1 = np.random.randn(N_hidden, 1)

        self.w2 = np.random.randn(1, N_hidden)  # / N_hidden
        self.b2 = np.random.randn(1, 1)

    def train_batch(self, x, y):
        """
            x shape =  [1024, mini_batch_size]
            y shape =  [1, mini_batch_size]
        """
        # layer 1 - input
        z1 = np.dot(self.w1, x) + self.b1
        a1 = f1(z1)

        # layer 2 - hidden
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = f2(z2)

        # layer 3 - output
        y_pred = vround(a2)  # np.vectorize(thresh)

        loss = np.mean(floss(a2, y))
        acc = 1 - np.sum(abs(y - y_pred)) / mini_batch_size

        self.backpropagate(x, y, z1, a1, z2, a2)

        return loss, acc

    def backpropagate(self, x, y, z1, a1, z2, a2):
        # general gradients
        dc_da2 = 2 * (y - a2)  # dC/da2
        da2_dz2 = df2(z2)  # da2/dz2
        dz2_da1 = self.w2.T  # dz2/da1
        da1_dz1 = df1(z1)  # da1/dz1

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
        self.w1 = self.w1 + dw1 * self.STEP_SIZE
        self.w2 = self.w2 + dw2 * self.STEP_SIZE
        self.b1 = self.b1 + db1 * self.STEP_SIZE
        self.b2 = self.b2 + db2 * self.STEP_SIZE

    def epoch(self, x_trn, y_trn):
        num_batch = int(len(y_trn) / mini_batch_size)

        loss, acc = [], []
        for i in range(num_batch):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            l, a = self.train_batch(x_trn[:, start:end], y_trn[start:end])
            loss.append(l), acc.append(a)
        if np.mean(loss) < 0.1: self.STEP_SIZE = 1e-3
        return mean(loss), mean(acc)

    def train_for(self):
        self.trn_data = randomize(*self.trn_data)
        loss, acc = self.epoch(*self.trn_data)
        print(f'epoch loss: {np.mean(loss):.5f}, epoch accuracy: {np.mean(acc):.5f}')
        return loss, acc

    def val_sample(self, x, y):
        # layer 1
        z1 = np.dot(self.w1, x) + self.b1
        a1 = f1(z1)

        # layer 2
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = f2(z2)

        # layer 3
        y_pred = f3(a2)

        loss = np.mean(floss(a2, y))
        acc = 1 - abs(a2 - y)
        return loss, acc

    def validate(self):
        loss, acc = [], []
        xs, ys = self.val_data

        # for x, y in zip(xs.T, ys):
        #     l, a = self.val_sample(x.reshape([len(x), 1]), y)
        #     if l < min_th: l = 0
        #     loss.append(l), acc.append(a)

        loss, acc = self.val_sample(xs, ys)

        print(f'total val loss: {np.mean(loss):.5f}, total val accuracy: {np.mean(acc):.5f}\n')
        return np.mean(loss), np.mean(acc)


if __name__ == "__main__":
    cc = CancerClassifier(*data_load(os.getcwd()))

    trn_loss, trn_acc, val_loss, val_acc = [], [], [], []
    for i in range(tot_ep):
        print(f"run number: {i + 1}/{tot_ep}")
        tl, ta = cc.train_for()
        vl, va = cc.validate()
        trn_loss.append(tl), trn_acc.append(ta), val_loss.append(vl), val_acc.append(va)

    plt.figure(), plt.plot(trn_loss, label='train'), plt.plot(val_loss, label='validation')
    plt.title("loss"), plt.legend(), plt.show()

    plt.figure(), plt.plot(trn_acc, label='train'), plt.plot(val_acc, label='validation')
    plt.title("acc"), plt.legend(), plt.show()
    # self = cc

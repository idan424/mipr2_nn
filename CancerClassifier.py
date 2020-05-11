from util import *

# initializing hyper parameters here
N_hidden = 12  # bigger is slower
mini_batch_size = 32  # bigger is faster
STEP_SIZE = 1e-3  # changes with respect to validation loss - see CancerClassifier.validate()
epochs = 600  # total number of epochs

# calling activation functions
f1, df1 = np.vectorize(leaky_relu), np.vectorize(dleaky_relu)  # acts on z1
f2, df2 = np.vectorize(sigmoid), np.vectorize(dsigmoid)  # acts on z2


class CancerClassifier(object):
    def __init__(self, x_trn, y_trn, x_val, y_val):
        self.STEP_SIZE = STEP_SIZE

        self.trn_data = x_trn, y_trn
        self.val_data = x_val, y_val

        self.w1 = np.random.randn(N_hidden, 1024)
        self.b1 = np.random.randn(N_hidden, 1)

        self.w2 = np.random.randn(1, N_hidden)
        self.b2 = np.random.randn(1, 1)

    def epoch(self):
        x_trn, y_trn = randomize(*self.trn_data)  # randomizing data order - <!><!><!>crucial for convergance<!><!><!>
        n_batch = int(y_trn.shape[0] / mini_batch_size)
        loss, acc = [], []

        # iterate over all training data
        for i in range(n_batch):
            start, end = i * mini_batch_size, (i + 1) * mini_batch_size
            l, a = self.train_batch(x_trn[:, start:end], y_trn[start:end])
            loss.append(l), acc.append(a)

        loss, acc = mean(loss), mean(acc)
        print(f'epoch loss: {loss:.5f}, epoch accuracy: {acc:.5f}')
        return loss, acc

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

        self.backpropagate(x, y, z1, a1, z2, a2)

        # computing batch loss and accuracy
        loss = np.mean(floss(y, a2))
        acc = 1 - np.sum(abs(y - np.vectorize(thresh)(a2))) / mini_batch_size
        return loss, acc

    def backpropagate(self, x, y, z1, a1, z2, a2):
        # general gradients (a, z)
        dc_da2 = dfloss(y, a2)  # dC/da2
        da2_dz2 = df2(z2)  # da2/dz2
        dz2_da1 = self.w2.T  # dz2/da1
        da1_dz1 = df1(z1)  # da1/dz1

        # layer specific gradients(w, b)
        dz2_dw2 = a1.T  # dz2/dw2
        dz2_db2 = np.ones([mini_batch_size, 1])  # dz2/db2
        dz1_dw1 = x.T  # dz1/dw1
        dz1_db1 = np.ones([mini_batch_size, 1])  # dz1/db1

        # layer 2 gradients
        dw2 = np.dot(dc_da2 * da2_dz2, dz2_dw2)  # dC/dw2 = dC/da2 * da2/dz2 * dz2/dw2
        db2 = np.dot(dc_da2 * da2_dz2, dz2_db2)  # dC/db2 = dC/da2 * da2/dz2 * dz2/db2

        # layer transfer gradient
        dc_da1 = np.dot(dz2_da1, dc_da2 * da2_dz2)

        # layer 1 gradients
        dw1 = np.dot(dc_da1 * da1_dz1, dz1_dw1)  # dC/dw1 = dC/da1 * da1/dz1 * dz1/dw1
        db1 = np.dot(dc_da1 * da1_dz1, dz1_db1)  # dC/db1 = dC/da1 * da1/dz1 * dz1/db1

        # actual descent
        self.update(dw1, db1, dw2, db2)

    def update(self, dw1, db1, dw2, db2):
        self.w1 += dw1 * self.STEP_SIZE
        self.w2 += dw2 * self.STEP_SIZE
        self.b1 += db1 * self.STEP_SIZE
        self.b2 += db2 * self.STEP_SIZE

    def validate(self):
        x, y = self.val_data

        # layer 1
        a1 = f1(np.dot(self.w1, x) + self.b1)

        # layer 2
        a2 = f2(np.dot(self.w2, a1) + self.b2)

        # computing loss and accuracy
        loss = np.mean(floss(a2, y))
        acc = np.mean(1 - abs(y - np.vectorize(round)(a2)))

        # loss-sensetive step size
        # self.step_correct(loss)    ->    it actually works better with no step correct

        print(f'total val loss: {loss:.5f}, total val accuracy: {acc:.5f}\n')
        return loss, acc

    def step_correct(self, loss):
        if loss < 0.15: self.STEP_SIZE = 5e-4
        if loss < 0.075: self.STEP_SIZE = 2.5e-4

    def to_json(self, va, filename=None):
        # this function saves a json version of the dict in to a file
        import json
        import datetime as dt
        trained_dict = {
            'weights': (self.w1.tolist(), self.w2.tolist()),
            'biases': (self.b1.tolist(), self.b2.tolist()),
            'nn_dim': N_hidden,
            'actication1': 'leaky_relu',
            'actication2': 'sigmoid',
            'IDs': ('305713034', '207127986')}
        if filename is None: filename = f'models/model_{dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}_val_acc_{va:.3f}'
        with open(filename, 'w') as f:
            json.dump(trained_dict, f)


def plotting(tl, ta, vl, va):
    # this plots the loss and accuracy
    plt.figure(), plt.subplot(211)
    plt.plot(tl, label='train'), plt.plot(vl, label='validation')
    plt.title("loss"), plt.legend(), plt.xlabel("epochs"), plt.ylabel("precents")
    plt.subplot(212)
    plt.plot(ta, label='train'), plt.plot(va, label='validation')
    plt.title("acc"), plt.legend(), plt.xlabel("epochs"), plt.ylabel("precents")
    plt.show()


def run_epochs(epochs, trn_loss, trn_acc, val_loss, val_acc):
    for e in range(epochs):
        print(f"run number: {e + 1}/{epochs}")
        tl, ta = cc.epoch()
        vl, va = cc.validate()
        trn_loss.append(tl), trn_acc.append(ta), val_loss.append(vl), val_acc.append(va)
    return trn_loss, trn_acc, val_loss, val_acc


if __name__ == "__main__":

    cc = CancerClassifier(*data_load(os.getcwd()))
    trn_loss, trn_acc, val_loss, val_acc = [], [], [], []
    trn_loss, trn_acc, val_loss, val_acc = run_epochs(epochs, trn_loss, trn_acc, val_loss, val_acc)
    # trn_loss, trn_acc, val_loss, val_acc = run_epochs(100, trn_loss, trn_acc, val_loss, val_acc)
    # cc.to_json(val_acc[-1], filename='fn')
    plotting(trn_loss, trn_acc, val_loss, val_acc)

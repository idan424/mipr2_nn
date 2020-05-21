from util import *

STEP_SIZE: float = 1e-3  # changes with respect to validation loss - see CancerClassifier.validate()
MIN_ACCURACY = 0.95  # minimum accuracy to save a model

# calling activation functions
f1, df1 = np.vectorize(leaky_relu), np.vectorize(dleaky_relu)  # acts on z1
f2, df2 = np.vectorize(sigmoid), np.vectorize(dsigmoid)  # acts on z2


class MSClassifier:
    __slots__ = ['STEP_SIZE', 'trn_data', 'val_data',
                 'w1', 'b1', 'w2', 'b2',
                 'loss_acc',
                 'best_dict', 'best_acc',
                 'batch_size', 'N_hidden']

    def __init__(self, x_trn: np.ndarray, y_trn: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                 bs: int, nh: int):
        """
        initializing an instance
        :param x_trn: training data
        :param y_trn: training labels
        :param x_val: validation data
        :param y_val: validation labels
        """
        self.STEP_SIZE = STEP_SIZE
        self.batch_size = bs
        self.N_hidden = nh

        self.trn_data = x_trn, y_trn
        self.val_data = x_val, y_val

        self.w1 = np.random.randn(1024, self.N_hidden)
        self.b1 = np.random.randn(1, self.N_hidden)

        self.w2 = np.random.randn(self.N_hidden, 1)
        self.b2 = np.random.randn(1, 1)

        self.loss_acc = [], [], [], []  # train_loss, train_acc, valid_loss, valid_acc

        self.best_dict, self.best_acc = {}, 0.0

    def epoch(self):
        """
        iterates over all training data once
        :return: loss, accuracy
        """
        x_trn, y_trn = randomize(*self.trn_data)  # randomizing data order - <!><!><!>crucial for convergance<!><!><!>
        n_batch = int(y_trn.shape[0] / self.batch_size)
        loss, acc = [], []

        for i in range(n_batch):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            l, a = self.train_batch(x_trn[start:end, :], y_trn[start:end])
            loss.append(l), acc.append(a)

        return mean(loss), mean(acc)

    def train_batch(self, x: np.ndarray, y: np.ndarray):
        """
        trains a mini batch of data
        :param x: array with shape [1024, mini_self.batch_size]
        :param y: array with shape [1, mini_self.batch_size]
        :return: loss, accuracy
        """
        # layer 1 - input
        z1 = np.dot(x, self.w1) + self.b1
        a1 = f1(z1)

        # layer 2 - hidden
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = f2(z2)

        self.backpropagate(x, y, z1, a1, z2, a2)

        # computing batch loss and accuracy
        loss = np.mean(floss(y, a2))
        acc = facc(y, a2)
        return loss, acc

    def backpropagate(self, x: np.ndarray, y: np.ndarray,
                      z1: np.ndarray, a1: np.ndarray,
                      z2: np.ndarray, a2: np.ndarray):
        """
        computes derivatives for weight update
        :param x: input data
        :param y: input labels
        :param z1: w1*x+b1
        :param a1: f(z1)
        :param z2: w2*a1+b2
        :param a2: f(z2)
        """
        # general gradients (a, z)
        dc_da2 = dfloss(y, a2)  # dC/da2
        da2_dz2 = df2(z2)  # da2/dz2
        dz2_da1 = self.w2.T  # dz2/da1
        da1_dz1 = df1(z1)  # da1/dz1

        # layer specific gradients(w, b)
        dz2_dw2 = a1.T  # dz2/dw2
        dz2_db2 = np.ones([1, self.batch_size])  # dz2/db2
        dz1_dw1 = x.T  # dz1/dw1
        dz1_db1 = np.ones([1, self.batch_size])  # dz1/db1

        # layer 2 gradients
        dw2 = np.dot(dz2_dw2, dc_da2 * da2_dz2)  # dC/dw2 = dC/da2 * da2/dz2 * dz2/dw2
        db2 = np.dot(dz2_db2, dc_da2 * da2_dz2)  # dC/db2 = dC/da2 * da2/dz2 * dz2/db2

        # layer transfer gradient
        dc_da1 = np.dot(dc_da2 * da2_dz2, dz2_da1)

        # layer 1 gradients
        dw1 = np.dot(dz1_dw1, dc_da1 * da1_dz1)  # dC/dw1 = dC/da1 * da1/dz1 * dz1/dw1
        db1 = np.dot(dz1_db1, dc_da1 * da1_dz1)  # dC/db1 = dC/da1 * da1/dz1 * dz1/db1

        # actual descent
        self.update(dw1, db1, dw2, db2)

    def update(self, dw1: np.ndarray, db1: np.ndarray, dw2: np.ndarray, db2: np.ndarray):
        """
        this actually does the update
        :param dw1: change in loss /w respect to w1
        :param db1: change in loss /w respect to b1
        :param dw2: change in loss /w respect to w2
        :param db2: change in loss /w respect to b2
        """
        self.w1 += dw1 * self.STEP_SIZE
        self.w2 += dw2 * self.STEP_SIZE
        self.b1 += db1 * self.STEP_SIZE
        self.b2 += db2 * self.STEP_SIZE

    def validate(self, save_best_flag: bool = False):
        """
        assesses the models accuracy and loss on validation data
        :return: loss, accuracy
        """
        x, y = self.val_data
        a1 = f1(np.dot(x, self.w1) + self.b1)  # layer 1
        a2 = f2(np.dot(a1, self.w2) + self.b2)  # layer 2
        # computing loss and accuracy
        loss = np.mean(floss(y, a2))
        acc = facc(y, a2)

        return loss, acc

    def to_json(self, acc: float = None, filename: str = None, trained_dict: dict = None):
        """
        saves a dictionary with relevant values
        :param trained_dict:
        :param acc: validation accuracy
        :param filename: save to a file named {filename}
        :param trained_dict: a dictionary of the trained model
        """
        # this function saves a json version of the dict in to a file
        import json
        import datetime as dt
        if acc is None:
            if self.best_acc > 0.89:
                acc = self.best_acc
            else:
                acc = self.loss_acc[3][-1]
        if trained_dict is None: trained_dict = self.to_dict()
        if filename is None:
            filename = f'models/model_{dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}' \
                       f'_val_acc_{acc:.3f}_(NN,bs)=({self.N_hidden},{self.batch_size}).json'
        with open(filename, 'w') as f:
            json.dump(trained_dict, f)

    def to_dict(self):
        return {
            'weights': (self.w1.tolist(), self.w2.tolist()),
            'biases': (self.b1.tolist(), self.b2.tolist()),
            'nn_dim': self.N_hidden,
            'actication1': 'leaky_relu',
            'actication2': 'sigmoid',
            'IDs': ('305713034', '207127986')}

    def run_epochs(self, ep: int, save_best_flag: bool = False, print_acc_loss: bool = False):
        """
        training on all the training data {epochs} times
        :param ep: number of training sessions on all of the data
        :param print_acc_loss:
        :param save_best_flag:
        """
        for i in range(ep):
            print(f"run number: {i + 1}/{ep}", end='\r')

            tl, ta = self.epoch()
            vl, va = self.validate(save_best_flag)

            if print_acc_loss:
                print(f'epoch loss: {tl:.5f}, epoch accuracy: {ta:.5f}')
                print(f'total val loss: {vl:.5f}, total val accuracy: {va:.5f}')

            self.loss_acc[0].append(tl), self.loss_acc[1].append(ta)
            self.loss_acc[2].append(vl), self.loss_acc[3].append(va)

            if save_best_flag and self.loss_acc[3][-1] > max(MIN_ACCURACY, self.best_acc):
                self.best_dict, self.best_acc = self.get_best_model_dict(), self.loss_acc[3][-1]
        print(f'epoch loss: {self.loss_acc[0][-1]:.5f}, epoch accuracy: {self.loss_acc[1][-1]:.5f}')
        print(f'total val loss: {self.loss_acc[2][-1]:.5f}, total val accuracy: {self.loss_acc[3][-1]:.5f}')
        if save_best_flag and self.best_acc >= 0.90: self.to_json(acc=self.best_acc, trained_dict=self.best_dict)
        print(f'best val accuracy: {self.best_acc:.5f}')

    def get_best_model_dict(self):
        if self.loss_acc[3][-1] > self.best_acc: return self.to_dict()


def plotting(tl: list, ta: list, vl: list, va: list):
    """
    plots the training and validation loss and accuracy along training
    :param tl: training loss
    :param ta: training accuracy
    :param vl: validation loss
    :param va: validation accuracy
    """
    plt.figure(), plt.subplot(211)
    plt.plot(tl, label='Train'), plt.plot(vl, label='Validation')
    plt.title("Loss"), plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Precents")
    plt.subplot(212)
    plt.plot(ta, label='Train'), plt.plot(va, label='Validation')
    plt.title("Accuracy"), plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Precents")
    plt.tight_layout(), plt.show()


if __name__ == "__main__":
    # initializing hyper parameters here
    epochs = 1000
    batch_size = 16
    N_hidden = 9
    msc = MSClassifier(*data_load(os.getcwd()), batch_size, N_hidden)
    msc.run_epochs(epochs, save_best_flag=True)
    plotting(*msc.loss_acc)
    # msc.run_epochs(100, save_best_flag=True)

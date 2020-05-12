import MSClassifier

N_Hiddens = range(8, 17)
batch_sizes = [16, 32, 64, 128]


class HPscanner:
    __slots__ = ['model']

    def __init__(self):
        self.model = MSClassifier

    # TODO: implement hyper-parameter scanner class
    def scan_hp(self):
        pass

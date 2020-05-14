from MSClassifier import *
import time
import sys

epochs = 1000
NUM_HP_EPOCH = 30  # a HP_EPOCH is an epoch in which all of the HP combinations have been scaned
# time took to complete 36 runs: 8 minutes and 27 seconds
# iterated over (N_hiddens = range(8, 17) @ batch_sizes = [16, 32, 64, 128])
N_hiddens = range(8, 17)
batch_sizes = [16, 32, 64, 128]
RUNS = len(N_hiddens) * len(batch_sizes) * NUM_HP_EPOCH


def check_dirs(models_dir):
    """
    checks that all necesarry dirs are present
    :param models_dir: the path to the models super directory
    :return: list of '.json' files in the models directory
    """
    ls = os.listdir(models_dir)
    dirs = [direc for direc in ls if '.json' not in direc]
    for nh in N_hiddens:
        for bs in batch_sizes:
            curr_dir = f'NN_{nh}_batch_{bs}'
            if curr_dir not in dirs:
                os.mkdir(f'{models_dir}\\{curr_dir}')
                print(f'created dir:{curr_dir}')
    return [file for file in ls if file not in dirs]


def sort_files():
    models_dir = f'{os.getcwd()}\\models'
    files = check_dirs(models_dir)

    for file in files:
        nh = int(file[file.find('(', 45) + 1:file.find(',', 45)])
        bs = int(file[file.find(',', 48) + 1:file.find(')', 48)])
        dirname = f'NN_{nh}_batch_{bs}'
        os.replace(f'{models_dir}\\{file}', f'{models_dir}\\{dirname}\\{file}')


def scan_hp(sleep=0, runs=0):
    run = 0
    for i in range(NUM_HP_EPOCH):
        for nh in N_hiddens:
            for bs in batch_sizes:
                run += 1
                print(f'HPrun: {run}/{runs} - NN_{nh}_batch_{bs} - HPepoch: {i + 1}/{NUM_HP_EPOCH}')
                mdl = MSClassifier(*data_load(os.getcwd()), bs, nh)  # data, batch_size, n_hidden
                mdl.run_epochs(epochs, save_best_flag=True)
                print('*' * 30)
                if sleep: time.sleep(sleep)
                sort_files()


def get_max_acc_file(direc):
    accs = []
    for file in os.listdir(direc):
        accs.append(float(file[file.find('acc_') + 4:file.find('acc') + 9]))
    if len(accs) > 0: return max(accs)
    return


def get_max_acc_dict(models_dir=f'{os.getcwd()}\\models'):
    dirs = os.listdir(models_dir)
    max_dict, best_dir_acc = {}, ("", 0)

    for direc in dirs:
        max_file_acc = get_max_acc_file(f'{models_dir}\\{direc}')
        if max_file_acc is not None and best_dir_acc[1] < max_file_acc:
            best_dir_acc = (direc, max_file_acc)
        max_dict[direc] = max_file_acc

    return max_dict, best_dir_acc


def train(slp, runs):
    print(f'starting {runs} runs')
    start = time.time()
    scan_hp(slp, runs)
    total = time.time() - (start + slp * runs)
    print(f'time took to run {runs} runs: {total:.3f} seconds')


if __name__ == '__main__':
    # train(slp=0, runs=RUNS)
    d, best = get_max_acc_dict()
    print(best)

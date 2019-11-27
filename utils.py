import os
import time


def prepar_data():
    if not os.path.exists('data'):
        os.mkdir('data')

    if not (os.path.exists('data/snli_1.0_train.txt')
            and os.path.exists('data/snli_1.0_dev.txt')
            and os.path.exists('data/snli_1.0_test.txt')):
        if not os.path.exists('data/snli_1.0.zip'):
            print('Downloading SNLI....')
            os.system('wget -P data https://nlp.stanford.edu/projects/snli/snli_1.0.zip')
        print('Unzipping SNLI....')
        os.system('unzip -d data/ data/snli_1.0.zip')
        os.system('mv -f data/snli_1.0/snli_*.txt data/')
    else:
        print('Found')


def get_current_time():
    return str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))[:-2]


def calc_eplased_time_since(start_time):
    curret_time = time.time()
    seconds = int(curret_time - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    time_str = '{:0>2d}h{:0>2d}min{:0>2d}s'.format(hours, minutes, seconds)
    return time_str
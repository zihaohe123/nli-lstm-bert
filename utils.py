import os


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
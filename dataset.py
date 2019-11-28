import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pickle
from transformers import BertTokenizer


class SNLIDataBert(Dataset):
    def __init__(self, args):
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.args = args
        self.train_data_path = os.path.join(self.args.data_path, 'snli_1.0_train.txt')
        self.dev_data_path = os.path.join(self.args.data_path, 'snli_1.0_dev.txt')
        self.test_data_path = os.path.join(self.args.data_path, 'snli_1.0_test.txt')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.init_data()

    def init_data(self):

        print('Initializing dev data')
        if os.path.exists(os.path.join(self.args.data_path, 'dev_data.pkl')):
            print('Found dev data')
            with open(os.path.join(self.args.data_path, 'dev_data.pkl'), 'rb') as f:
                self.dev_data = pickle.load(f)
        else:
            self.dev_data = self.load_data(self.dev_data_path)
            with open(os.path.join(self.args.data_path, 'dev_data.pkl'), 'wb') as f:
                pickle.dump(self.dev_data, f)

        print('Initializing test data')
        if os.path.exists(os.path.join(self.args.data_path, 'test_data.pkl')):
            print('Found test data')
            with open(os.path.join(self.args.data_path, 'test_data.pkl'), 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            self.test_data = self.load_data(self.test_data_path)
            with open(os.path.join(self.args.data_path, 'test_data.pkl'), 'wb') as f:
                pickle.dump(self.test_data, f)

        print('Initializing train data')
        if os.path.exists(os.path.join(self.args.data_path, 'train_data.pkl')):
            print('Found train data')
            with open(os.path.join(self.args.data_path, 'train_data.pkl'), 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = self.load_data(self.train_data_path)
            with open(os.path.join(self.args.data_path, 'train_data.pkl'), 'wb') as f:
                pickle.dump(self.train_data, f)

    def load_data(self, path):
        print('Loading data....{}'.format(path))
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for idx, line in enumerate(f):

                # skip the first line
                if idx == 0:
                    continue
                if idx % 5000 == 0:
                    print('{}'.format(idx))

                cols = line.strip('\n').split('\t')

                #   '–' indicates a lack of consensus from the human annotators, ignore it
                if cols[0] == '-':
                    continue

                premise, hypothesis = cols[5], cols[6]
                premise_ids = self.tokenizer.encode(premise)
                hypothesis_ids = self.tokenizer.encode(hypothesis)
                pair_token_ids = [101] + premise_ids + [102] + hypothesis_ids + [102]  # 101-->[CLS], 102-->[SEP]. This is the format of sentence-pair embedding for BERT
                premise_len = len(premise_ids)  # the length does not consider the added SEP in the end
                hypothesis_len = len(hypothesis_ids)
                segment_ids = torch.tensor(
                    [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
                attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

                token_ids.append(torch.tensor(pair_token_ids))
                seg_ids.append(segment_ids)
                mask_ids.append(attention_mask_ids)
                y.append(self.label_dict[cols[0]])

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        print(len(dataset))
        return dataset

    def get_train_dev_loader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        dev_loader = DataLoader(
            self.dev_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, dev_loader

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        dev_loader = DataLoader(
            self.dev_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, dev_loader, test_loader

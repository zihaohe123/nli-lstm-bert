import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from model import MatchLSTM
from dataset import SNLIDataBert
from utils import prepar_data
from transformers import AdamW


class Solver:
    def __init__(self, args):
        # how to use GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_workers = max([4 * torch.cuda.device_count(), 4])

        torch.manual_seed(args.seed)
        prepar_data()

        # prepare data
        snli_dataset = SNLIDataBert(args)
        train_loader, dev_loader, test_loader = snli_dataset.get_dataloaders(batch_size=args.batch_size,
                                                                             num_workers=num_workers,
                                                                             pin_memory=device == 'cuda')
        print('#examples:',
              '#train', len(train_loader.dataset),
              '#dev', len(dev_loader.dataset),
              '#test', len(test_loader.dataset))

        model = MatchLSTM(args)

        device_count = 0
        if device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                model = nn.DataParallel(model, dim=1)
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(device_count))
        model.to(device)

        # LSTM optimizer
        params = model.module.req_grad_params if device_count > 1 else model.req_grad_params
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), amsgrad=True)

        # Bert optimizer
        param_optimizer = list(model.bert.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer_bert = AdamW(optimizer_grouped_parameters, lr=2e-5)

        loss_func = nn.CrossEntropyLoss()
        args.name += '_bert' if args.train_bert else ''
        ckpt_path = os.path.join('ckpt', '{}.pth'.format(args.name))
        if not os.path.exists(ckpt_path):
            print('Not found ckpt', ckpt_path)

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.optimizer_bert = optimizer_bert
        self.loss_func = loss_func
        self.device = device
        self.snli_dataset = snli_dataset
        self.ckpt_path = ckpt_path
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

    def train(self):
        print('Starting Traing....')
        best_loss = float('inf')
        best_acc = 0.
        best_epoch = 0

        for epoch in range(1, self.args.epochs + 1):
            print('-'*40 + 'Epoch: {}'.format(epoch) + '-'*40)
            self.train_epoch(epoch)
            dev_loss, dev_acc = self.evaluate_epoch(epoch, 'Dev')
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_acc = dev_acc
                best_epoch = epoch
                self.save_model()

            print('{}\tLowest Dev Loss {:.6f}, '
                  'Best Dev Acc. {:.1f}%, '
                  'Epoch {}'.format(datetime.now(), best_loss, 100 * best_acc, best_epoch))

            self.evaluate_epoch(epoch, 'Test')

            # Learning rate decay
            for param_group in self.optimizer.param_groups:
                print('lr: {:.6f} -> {:.6f}'.format(param_group['lr'], param_group['lr'] * self.args.lr_decay))
                param_group['lr'] *= self.args.lr_decay

        self.test()

    def test(self):
        # Load the best checkpoint
        self.load_model()

        # Test
        print('Final result..............')
        self.evaluate_epoch(self.args.epochs, 'Test')

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.
        example_count = 0
        correct = 0
        start_t = datetime.now()
        for batch_idx, (pair_token_ids, premise_lens, hypothesis_lens, mask_ids, seg_ids, y) in enumerate(self.train_loader):
            pair_token_ids = pair_token_ids.to(self.device)
            mask_ids = mask_ids.to(self.device)
            seg_ids = seg_ids.to(self.device)
            target = y.to(self.device)
            output = self.model(pair_token_ids, premise_lens, hypothesis_lens, mask_ids, seg_ids)
            self.optimizer.zero_grad()
            self.optimizer_bert.zero_grad()
            loss = self.loss_func(output, target)
            loss.backward()
            if self.args.grad_max_norm > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.req_grad_params, self.args.grad_max_norm)
            self.optimizer.step()
            self.optimizer_bert.step()

            batch_loss = len(output) * loss.item()
            train_loss += batch_loss
            example_count += len(target)

            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            if (batch_idx + 1) % self.args.log_interval == 0 \
                    or batch_idx == len(self.train_loader) - 1:
                _progress = '{}\t' \
                            'Train Epoch {}/{}, ' \
                            '[{}/{} ({:.1f}%)],' \
                            ' Batch Loss: {:.6f}'.format(datetime.now(),
                                                         epoch, self.args.epochs,
                                                         example_count, len(self.train_loader.dataset),
                                                         100. * example_count / len(self.train_loader.dataset),
                                                         batch_loss / len(output))
                print(_progress)

        train_loss /= len(self.train_loader.dataset)
        acc = correct / len(self.train_loader.dataset)
        print('{}\tTrain Epoch {}, Avg Train Loss: {:.6f}, Train Accuracy: {}/{} ({:.1f}%)'.
              format(datetime.now() - start_t, epoch, train_loss,
                     correct, len(self.train_loader.dataset), 100. * acc))
        return train_loss

    def evaluate_epoch(self, epoch, mode):
        self.model.eval()
        if mode == 'Dev':
            loader = self.dev_loader
        else:
            loader = self.test_loader
        eval_loss = 0.
        correct = 0
        start_t = datetime.now()
        with torch.no_grad():
            for batch_idx, (pair_token_ids, premise_lens, hypothesis_lens, mask_ids, seg_ids, y) in enumerate(loader):
                pair_token_ids = pair_token_ids.to(self.device)
                mask_ids = mask_ids.to(self.device)
                seg_ids = seg_ids.to(self.device)
                target = y.to(self.device)
                output = self.model(pair_token_ids, premise_lens, hypothesis_lens, mask_ids, seg_ids)
                loss = self.loss_func(output, target)
                eval_loss += len(output) * loss.item()
                pred = torch.max(output, 1)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        eval_loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)
        print('{}\t{} Epoch {}, Avg {} Loss: {:.6f}, '
              '{} Accuracy: {}/{} ({:.1f}%)'.format(datetime.now() - start_t, mode,
                                                 epoch, mode, eval_loss,
                                                 mode, correct, len(loader.dataset),
                                                 100. * acc))
        return eval_loss, acc

    def save_model(self):
        # save a model and args
        model_dict = dict()
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['m_config'] = self.args
        model_dict['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(os.path.dirname(self.ckpt_path)):
            os.makedirs(os.path.dirname(self.ckpt_path))
        torch.save(model_dict, self.ckpt_path)
        print('Saved', self.ckpt_path)

    def load_model(self):
        print('Load checkpoint', self.ckpt_path)
        checkpoint = torch.load(self.ckpt_path)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            self.model.module.load_state_dict(checkpoint['state_dict'])
import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import MatchLSTM
from dataset import SNLIDataBert
from utils import prepar_data
from transformers import AdamW
import time
from utils import get_current_time, calc_eplased_time_since


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
                model = nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(device_count))
        model.to(device)

        # LSTM optimizer
        params = model.module.req_grad_params if device_count > 1 else model.req_grad_params
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), amsgrad=True)

        # Bert optimizer
        param_optimizer = list(model.module.bert.named_parameters() if device_count > 1 else model.bert.named_parameters())
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

        batches = len(train_loader.dataset) // args.batch_size
        log_interval = batches // 30

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
        self.log_interval = log_interval

    def train(self):
        print('Starting Traing....')
        best_loss = float('inf')
        best_acc = 0.
        best_epoch = 0

        train_start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            print('-'*20 + 'Epoch: {}, {}'.format(epoch, get_current_time()) + '-'*20)
            train_loss, train_acc = self.train_epoch()
            dev_loss, dev_acc = self.evaluate_epoch('Dev')
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_acc = dev_acc
                best_epoch = epoch
                self.save_model()

            print('Epoch: {:0>2d}/{}\n'
                  'Epoch Training Time: {}\n'
                  'Elapsed Time: {}\n'
                  'Train Loss: {:.3f}, Train Acc: {:.3f}\n'
                  'Dev Loss: {:.3f}, Dev Acc: {:.3f}\n'
                  'Best Dev Loss: {:.3f}, Best Dev Acc: {:.3f}, '
                  'Best Dev Acc Epoch: {:0>2d}\n'.format(epoch, self.args.epochs,
                                                       calc_eplased_time_since(epoch_start_time),
                                                       calc_eplased_time_since(train_start_time),
                                                       train_loss, train_acc,
                                                       dev_loss, dev_acc,
                                                       best_loss, best_acc, best_epoch))

            # LSTM learning rate decay
            for param_group in self.optimizer.param_groups:
                print('lr: {:.6f} -> {:.6f}\n'.format(param_group['lr'], param_group['lr'] * self.args.lr_decay))
                param_group['lr'] *= self.args.lr_decay

        print('Training Finished!')

        self.test()

    def test(self):
        # Load the best checkpoint
        self.load_model()

        # Test
        print('Final result..............')
        test_loss, test_acc = self.evaluate_epoch('Test')
        print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))

    def train_epoch(self):
        self.model.train()
        train_loss = 0.
        example_count = 0
        correct = 0
        batch_start_time = time.time()
        for batch_idx, (pair_token_ids, premise_lens, hypothesis_lens, mask_ids, seg_ids, y) in enumerate(self.train_loader):
            pair_token_ids = pair_token_ids.to(self.device)
            mask_ids = mask_ids.to(self.device)
            seg_ids = seg_ids.to(self.device)
            target = y.to(self.device)
            premise_lens = premise_lens.unsqueeze(1)    # batchsize --> batchsize*1 for data parallel
            hypothesis_lens = hypothesis_lens.unsqueeze(1)  # batchsize --> batchsize*1 for data parallel
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

            if batch_idx == 0 or (batch_idx+1) % self.log_interval == 0 or batch_idx+1 == self.log_interval:
                print('Batch: {:0>5d}/{:0>5d}, '
                      'Batch Training Time: {}, '
                      'Batch Loss: {:.3f}'.format(batch_idx+1, len(self.train_loader),
                                                  calc_eplased_time_since(batch_start_time),
                                                  batch_loss / len(output)))
                batch_start_time = time.time()

        train_loss /= len(self.train_loader.dataset)
        acc = correct / len(self.train_loader.dataset)
        print()
        return train_loss, acc

    def evaluate_epoch(self, mode):
        print('Evaluating....')
        self.model.eval()
        if mode == 'Dev':
            loader = self.dev_loader
        else:
            loader = self.test_loader
        eval_loss = 0.
        correct = 0
        with torch.no_grad():
            for batch_idx, (pair_token_ids, premise_lens, hypothesis_lens, mask_ids, seg_ids, y) in enumerate(loader):
                pair_token_ids = pair_token_ids.to(self.device)
                mask_ids = mask_ids.to(self.device)
                seg_ids = seg_ids.to(self.device)
                target = y.to(self.device)
                premise_lens = premise_lens.unsqueeze(1)    # batchsize --> batchsize*1 for data parallel
                hypothesis_lens = hypothesis_lens.unsqueeze(1)  # batchsize --> batchsize*1 for data parallel
                output = self.model(pair_token_ids, premise_lens, hypothesis_lens, mask_ids, seg_ids)
                loss = self.loss_func(output, target)
                eval_loss += len(output) * loss.item()
                pred = torch.max(output, 1)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        eval_loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)
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
        print()

    def load_model(self):
        print('Load checkpoint', self.ckpt_path)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            # if saving a paralleled model but loading an unparalleled model
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['state_dict'])
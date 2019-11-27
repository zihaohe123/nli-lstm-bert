import argparse
import pprint


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="mLSTM")

    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--premise_max_len', type=int, default=83)
    parser.add_argument('--hypothesis_max_len', type=int, default=62)
    parser.add_argument('--train_bert', action='store_true', default=False, help='Whether to fine-tune bert')

    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--grad_max_norm', type=float, default=0.)  #

    parser.add_argument('--dropout_fc', type=float, default=0.)  #
    parser.add_argument('--dropout_emb', type=float, default=0.3)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--log_interval', type=int, default=500, help='Print results on every log_interval batches when training')
    parser.add_argument('--gpu', type=str, default='', help='which gpus to use')
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)
    return args
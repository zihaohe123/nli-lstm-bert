import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import BertModel


class MatchLSTM(nn.Module):
    def __init__(self, config):
        super(MatchLSTM, self).__init__()
        self.config = config

        use_cuda = config.gpu != '' and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # TODO emb_partial update
        # https://github.com/shuohangwang/SeqMatchSeq/blob/master/main/main.lua#L42

        self.fc_bert = nn.Linear(in_features=768, out_features=config.num_classes)
        self.dropout_emb = nn.Dropout(p=config.dropout_emb)

        # Bert parameters not included because we haven't deifined BERT yet
        self.req_grad_params = self.get_req_grad_params(debug=False)

        # BERT Model. We use a pre-trained one.
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # if not fine-tuning Bert, we freeze its gradients
        if not config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def init_linears(self):
        nn.init.uniform_(self.w_e)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, pair, mask_id, seg_id):

        # feed the pair token ids into BertModel
        pair = self.bert(pair, token_type_ids=seg_id, attention_mask=mask_id)[0]
        pair = self.dropout_emb(pair)

        h = pair[:, 0]
        return self.fc_bert(h)


    def get_req_grad_params(self, debug=False):
        print('#LSTM parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())  # the product of all dimensions, i.e., # of parameters
                total_size += n_params
            if debug:
                print(name, p.requires_grad, p.size(), multiply_iter(p.size()), sep='\t')
        print('{:,}'.format(total_size))
        return params

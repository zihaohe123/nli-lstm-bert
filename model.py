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

        # BERT Model. We use a pre-trained one.
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        if not config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.w_e = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=True)
        self.w_s = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.w_t = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.w_m = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.fc = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)
        self.init_linears()

        self.lstm_prem = nn.LSTM(768, config.hidden_size)   # 768 is the embedding dim of BERT
        self.lstm_hypo = nn.LSTM(768, config.hidden_size)   # 768 is the embedding dim of BERT
        self.lstm_match = nn.LSTMCell(2 * config.hidden_size, config.hidden_size)

        self.dropout_fc = nn.Dropout(p=config.dropout_fc)
        self.dropout_emb = nn.Dropout(p=config.dropout_emb)

        self.req_grad_params = self.get_req_grad_params(debug=False)

    def init_linears(self):
        nn.init.uniform_(self.w_e)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, pair, premise_len, hypothesis_len, mask_id, seg_id):
        batch_size = pair.shape[0]

        # feed the pair token ids into BertModel
        pair = self.bert(pair, token_type_ids=seg_id, attention_mask=mask_id)[0]
        pair = self.dropout_emb(pair)
        premise = [torch.tensor(pair[i][1:2+premise_len[i]]) for i in range(batch_size)]    # including the end [SEP]
        hypothesis = [torch.tensor(pair[i][2+premise_len[i]: 2+premise_len[i]+hypothesis_len[i]]) for i in range(batch_size)]

        premise = pad_sequence(premise, batch_first=True)
        hypothesis = pad_sequence(hypothesis, batch_first=True)

        # premise
        prem_max_len = premise.shape[1]
        premise_len += 1  # we add 1 for the ending [SEP]. This is only for the premise but not the hypothesis
        premise_len, p_idxes = torch.sort(premise_len, descending=True)
        _, p_idx_unsort = torch.sort(p_idxes)      # in order to restore the original order
        premise = premise[p_idxes]
        packed_premise = pack_padded_sequence(premise, premise_len, batch_first=True)
        # (max_len, batch_size, hidden_size)
        h_s, (_, _) = self.lstm_prem(packed_premise)
        h_s, _ = pad_packed_sequence(h_s)
        h_s = h_s[:, p_idx_unsort]  # because we have two sentences here, we need to restore the order to ensuring matching

        # hypothesis
        # hypothesis = hypothesis.to(self.device)
        hypothesis_max_len = hypothesis.shape[1]
        hypothesis_len, h_idxes = torch.sort(hypothesis_len, descending=True)
        _, h_idx_unsort = torch.sort(h_idxes)
        hypothesis = hypothesis[h_idxes]
        packed_hypothesis = pack_padded_sequence(hypothesis, hypothesis_len, batch_first=True)
        # (max_len, batch_size, hidden_size)
        h_t, (_, _) = self.lstm_hypo(packed_hypothesis)
        h_t, _ = pad_packed_sequence(h_t)
        h_t = h_t[:, h_idx_unsort]
        hypothesis_len = hypothesis_len[h_idx_unsort]  # because we have two sentences here, we need to restore the order to ensuring matching

        # matchLSTM. This is the core of this paper.
        batch_size = premise.shape[0]
        h_m_k = torch.zeros((batch_size, self.config.hidden_size), device=self.device)
        c_m_k = torch.zeros((batch_size, self.config.hidden_size), device=self.device)
        h_last = torch.zeros((batch_size, self.config.hidden_size), device=self.device)

        for k in range(hypothesis_max_len):
            h_t_k = h_t[k]

            # Equation (6)
            # e_kj: (prem_max_len, batch_size)
            e_kj = torch.zeros((prem_max_len, batch_size), device=self.device)
            w_e_expand = self.w_e.expand(batch_size, self.config.hidden_size)
            for j in range(prem_max_len):
                # tanh_stm: (batch_size, hidden_size)
                tanh_s_t_m = torch.tanh(self.w_s(h_s[j]) + self.w_t(h_t_k) + self.w_m(h_m_k))

                # dot product
                # https://github.com/pytorch/pytorch/issues/18027#issuecomment-473404765
                e_kj[j] = (w_e_expand * tanh_s_t_m).sum(-1)

            # Equation (3)
            # (prem_max_len, batch_size)
            alpha_kj = F.softmax(e_kj, dim=0)

            # Equation (2)
            # (batch_size, hidden_size)
            a_k = torch.bmm(torch.unsqueeze(alpha_kj.t(), 1), h_s.permute(1, 0, 2))
            a_k = torch.squeeze(a_k, dim=1)

            # Equation (7)
            # (batch_size, 2 * hidden_size)
            m_k = torch.cat((a_k, h_t_k), 1)

            # Equation (8)
            # (batch_size, hidden_size)
            h_m_k, c_m_k = self.lstm_match(m_k, (h_m_k, c_m_k))

            # handle variable length sequences: hypothesis
            # (batch_size)
            for batch_idx, hl in enumerate(hypothesis_len):
                if k + 1 == hl:
                    h_last[batch_idx] = h_m_k[batch_idx]

        h_last = self.dropout_fc(h_last)

        return self.fc(h_last)

    def get_req_grad_params(self, debug=False):
        print('#parameters: ', end='')
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

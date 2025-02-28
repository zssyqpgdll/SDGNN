from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random
from ggnn import GGNNModel

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                torch.nn.init.xavier_uniform_(wt.data)
                # wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    torch.nn.init.xavier_normal_(linear.weight.data)
    # linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    torch.nn.init.xavier_normal_(wt.data)
    # wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    torch.nn.init.xavier_uniform_(wt.data)
    # wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ggnn = GGNNModel()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        init_linear_wt(self.W_h)
        self.lstm = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim, bidirectional=True, batch_first=True)
        self.graph_enc_outputs_W_h = nn.Linear(config.hidden_dim, config.hidden_dim * 2, bias=False)
        init_linear_wt(self.graph_enc_outputs_W_h)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens, enc_graph_batch, enc_graph_mask_batch, enc_node_type_mask_batch):
        embedded = self.embedding(input)
        adjacency_transpose_matrix = enc_graph_batch.permute(1, 0, 2, 3).to(torch.float32)
        adjacency_transpose_matrix_mask = enc_graph_mask_batch.permute(1, 0, 2, 3).to(torch.float32)
        node_type_mask = enc_node_type_mask_batch.permute(1, 0, 2, 3).to(torch.float32)

        graph_enc_outputs = self.ggnn(embedded, seq_lens, adjacency_transpose_matrix, adjacency_transpose_matrix_mask, node_type_mask, is_ast=False) # [b, s, h]
        graph_dec_in_state = self.ggnn.reduce_final_state_with_max(graph_enc_outputs) # Tuple([b, h], [b, h])
        graph_enc_outputs = self.graph_enc_outputs_W_h(graph_enc_outputs)

        ast_enc_outputs = self.ggnn(embedded, seq_lens, adjacency_transpose_matrix, adjacency_transpose_matrix_mask, node_type_mask, is_ast=True) # [b, s, h]
        packed = pack_padded_sequence(ast_enc_outputs, seq_lens, batch_first=True, enforce_sorted=False)
        output, (fw_st, bw_st) = self.lstm(packed) # fw_st:[2, b, h]
        ast_encoder_outputs, _ = pad_packed_sequence(output, batch_first=True) # [b, s, 2*h]
        ast_dec_in_state = (fw_st, bw_st) # Tuple([2, b, h], [2, b, h])

        graph_encoder_outputs_new = graph_enc_outputs.contiguous()
        graph_encoder_feature = graph_encoder_outputs_new.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        graph_encoder_feature = self.W_h(graph_encoder_feature)

        ast_encoder_outputs_new = ast_encoder_outputs.contiguous()
        ast_encoder_feature = ast_encoder_outputs_new.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        ast_encoder_feature = self.W_h(ast_encoder_feature)

        return graph_enc_outputs, ast_encoder_outputs, graph_encoder_feature, ast_encoder_feature, \
            graph_dec_in_state, ast_dec_in_state

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, graph_hidden, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        h_in = torch.cat([h_in, graph_hidden[0]], dim=1)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        c_in = torch.cat([c_in, graph_hidden[1]], dim=1)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 6 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 5, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)


    def forward(self, y_t_1, s_t_1, graph_enc_outputs, ast_encoder_outputs, graph_encoder_feature, ast_encoder_feature,
                enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            graph_c_t, _, coverage_next = self.attention_network(s_t_hat, graph_enc_outputs, graph_encoder_feature,
                                                              enc_padding_mask, coverage)
            ast_c_t, _, coverage_next = self.attention_network(s_t_hat, ast_encoder_outputs, ast_encoder_feature,
                                                           enc_padding_mask, coverage)
            c_t = torch.cat([graph_c_t, ast_c_t], dim=1)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        graph_c_t, graph_attn_dist, coverage_next = self.attention_network(s_t_hat, graph_enc_outputs, graph_encoder_feature,
                                                          enc_padding_mask, coverage)
        ast_c_t, ast_attn_dist, coverage_next = self.attention_network(s_t_hat, ast_encoder_outputs, ast_encoder_feature,
                                                           enc_padding_mask, coverage) #[b, 2*h]
        c_t = torch.cat([graph_c_t, ast_c_t], dim=1) #[b, 2*2*h]
        attn_dist = graph_attn_dist + ast_attn_dist
        attn_dist = attn_dist / torch.sum(attn_dist, dim=1, keepdim=True)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + 2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 5
        output = self.out1(output) # B x hidden_dim

        output = F.tanh(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

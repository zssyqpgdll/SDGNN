#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from data_util import config
import numpy as np


# 加载node的节点类型字典
with open('./node_type.json', 'r') as f:
    node_type_vocab_dict = json.load(f)

# 加载graph的边类型字典
with open('./edge_vocab.json', 'r') as f:
    edge_vocab_dict = json.load(f)

class GGNNModel(nn.Module):
    def __init__(self):
        super(GGNNModel, self).__init__()
        self.num_edge_types = len(edge_vocab_dict)
        self.all_num_edge_types = 2 * self.num_edge_types

        self.trunc_norm_init = torch.nn.init.xavier_uniform_

        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.batch_size = config.batch_size


        if self.emb_dim != self.hidden_dim:
            self._first_W = nn.Parameter(torch.empty(self.emb_dim, self.hidden_dim))
            self.trunc_norm_init(self._first_W)  # Initialize with Xavier uniform initializer
            self._first_b = nn.Parameter(torch.empty(1, self.hidden_dim))
            self.trunc_norm_init(self._first_b)

        # Reset gate
        self._reset_W_wl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(self.hidden_dim, self.hidden_dim))) for _ in range(self.all_num_edge_types)])
        self._reset_W_bl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(1, self.hidden_dim))) for _ in range(self.all_num_edge_types)])
        self._reset_U_wl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(self.hidden_dim, self.hidden_dim))) for _ in range(len(node_type_vocab_dict))])
        self._reset_U_bl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(1, self.hidden_dim))) for _ in range(len(node_type_vocab_dict))])

        # Update gate
        self._update_W_wl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(self.hidden_dim, self.hidden_dim))) for _ in range(self.all_num_edge_types)])
        self._update_W_bl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(1, self.hidden_dim))) for _ in range(self.all_num_edge_types)])
        self._update_U_wl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(self.hidden_dim, self.hidden_dim))) for _ in range(len(node_type_vocab_dict))])
        self._update_U_bl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(1, self.hidden_dim))) for _ in range(len(node_type_vocab_dict))])

        # Main transformation, using label-wise parameters
        self._W_wl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(self.hidden_dim, self.hidden_dim))) for _ in range(self.all_num_edge_types)])
        self._W_bl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(1, self.hidden_dim))) for _ in range(self.all_num_edge_types)])
        self._U_wl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(self.hidden_dim, self.hidden_dim))) for _ in range(len(node_type_vocab_dict))])
        self._U_bl = nn.ParameterList(
            [nn.Parameter(self.trunc_norm_init(torch.empty(1, self.hidden_dim))) for _ in range(len(node_type_vocab_dict))])


    def forward(self, hidden_inputs, seq_len, adjacency_transpose_matrix, adjacency_transpose_matrix_mask, node_type_mask, is_ast = False):
        self.adjacency_transpose_matrix = adjacency_transpose_matrix
        self.adjacency_transpose_matrix_mask = adjacency_transpose_matrix_mask
        self.node_type_mask = node_type_mask
        self.is_ast = is_ast

        self.max_seq_len = np.max(seq_len)
        hidden_inputs = hidden_inputs.view(self.batch_size * self.max_seq_len, config.emb_dim)
        if config.emb_dim != config.hidden_dim:
            hidden_inputs = torch.matmul(hidden_inputs, self._first_W)
            hidden_inputs = hidden_inputs + self._first_b

        for time_step in range(config.gru_times_step):
            self.r = self._compute_reset(hidden_inputs) # [b, v, h]
            self.u = self._compute_update(hidden_inputs) # [b, v, h]
            self.h_hat = self._compute_h_hat(hidden_inputs) # [b, v, h]

            hidden_inputs = (1 - self.u) * hidden_inputs.view(self.batch_size, self.max_seq_len, self.hidden_dim) + self.u * self.h_hat # [b, v, h]
            hidden_inputs = hidden_inputs.view(self.batch_size * self.max_seq_len, self.hidden_dim)  # [b*v, h]

        outputs = hidden_inputs.view(self.batch_size, self.max_seq_len, self.hidden_dim)
        return outputs


    def _compute_reset(self, hidden_inputs):
        W_acts, U_acts = None, None

        for edge_type in range(self.all_num_edge_types):
            if self.is_ast and edge_type != 0 and edge_type != self.num_edge_types:
                continue
            # linear transformation
            reset_W_wi = self._reset_W_wl[edge_type]
            reset_W_bi = self._reset_W_bl[edge_type]
            outputs = torch.matmul(hidden_inputs, reset_W_wi) # [b*v, h]
            outputs = outputs + reset_W_bi # [b*v, h]

            outputs = outputs.view(self.batch_size, self.max_seq_len, self.hidden_dim)

            if self.is_ast:
                if edge_type == 0:
                    for l in range(0, config.max_layer + 1):
                        outputs = torch.matmul(self.adjacency_transpose_matrix_mask[l] * self.adjacency_transpose_matrix[edge_type], outputs) # [b, v, h]
                else:
                    for l in range(config.max_layer, -1, -1):
                        outputs = torch.matmul(self.adjacency_transpose_matrix_mask[l] * self.adjacency_transpose_matrix[edge_type], outputs) # [b, v, h]
            else:
                outputs = torch.matmul(self.adjacency_transpose_matrix[edge_type], outputs)

            if edge_type==0:
                W_acts = outputs
            else:
                W_acts += outputs

        for node_type in range(len(node_type_vocab_dict)):
            reset_U_wi = self._reset_U_wl[node_type]
            reset_U_bi = self._reset_U_bl[node_type]
            outputs = torch.matmul(hidden_inputs, reset_U_wi) # [b*v, h]
            outputs = outputs + reset_U_bi # [b*v, h]

            outputs = torch.reshape(outputs, [self.batch_size, self.max_seq_len, self.hidden_dim])  # [b, v, h]

            if node_type==0:
                U_acts = outputs * self.node_type_mask[node_type] # [b, v, h]
            else:
                U_acts += outputs * self.node_type_mask[node_type] # [b, v, h]

        reset_gate = torch.sigmoid(W_acts + U_acts)
        return reset_gate


    def _compute_update(self, hidden_inputs):
        W_acts, U_acts = None, None

        for edge_type in range(self.all_num_edge_types):
            if self.is_ast and edge_type != 0 and edge_type != self.num_edge_types:
                continue
            # linear transformation
            update_W_wi = self._update_W_wl[edge_type]
            update_W_bi = self._update_W_bl[edge_type]
            outputs = torch.matmul(hidden_inputs, update_W_wi) # [b*v, h]
            outputs = outputs + update_W_bi # [b*v, h]

            outputs = outputs.view(self.batch_size, self.max_seq_len, self.hidden_dim)  # [b, v, h]

            if self.is_ast:
                if edge_type == 0:
                    for l in range(0, config.max_layer + 1):
                        outputs = torch.matmul(self.adjacency_transpose_matrix_mask[l] * self.adjacency_transpose_matrix[edge_type], outputs)  # [b, v, h]
                else:
                    for l in range(config.max_layer, -1, -1):
                        outputs = torch.matmul(self.adjacency_transpose_matrix_mask[l] * self.adjacency_transpose_matrix[edge_type], outputs)  # [b, v, h]
            else:
                outputs = torch.matmul(self.adjacency_transpose_matrix[edge_type], outputs)

            if edge_type == 0:
                W_acts = outputs
            else:
                W_acts += outputs


        for node_type in range(len(node_type_vocab_dict)):
            update_U_wi = self._update_U_wl[node_type]
            update_U_bi = self._update_U_bl[node_type]
            outputs = torch.matmul(hidden_inputs, update_U_wi) # [b*v, h]
            outputs = outputs + update_U_bi # [b*v, h]

            outputs = outputs.view(self.batch_size, self.max_seq_len, self.hidden_dim)  # [b, v, h]

            if node_type==0:
                U_acts = outputs * self.node_type_mask[node_type] # [b, v, h]
            else:
                U_acts += outputs * self.node_type_mask[node_type] # [b, v, h]

        update_gate = torch.sigmoid(W_acts + U_acts)
        return update_gate


    def _compute_h_hat(self, hidden_inputs):
        W_acts, U_acts = None, None

        for edge_type in range(self.all_num_edge_types):
            if self.is_ast and edge_type != 0 and edge_type != self.num_edge_types:
                continue
            # linear transformation
            W_wi = self._W_wl[edge_type]
            W_bi = self._W_bl[edge_type]
            outputs = torch.matmul(hidden_inputs, W_wi) # [b*v, h]
            outputs = outputs + W_bi # [b*v, h]

            outputs = outputs.view(self.batch_size, self.max_seq_len, self.hidden_dim)  # [b, v, h]

            if self.is_ast:
                if edge_type == 0:
                    for l in range(0, config.max_layer + 1):
                        outputs = torch.matmul(self.adjacency_transpose_matrix_mask[l] * self.adjacency_transpose_matrix[edge_type], outputs)  # [b, v, h]
                else:
                    for l in range(config.max_layer, -1, -1):
                        outputs = torch.matmul(self.adjacency_transpose_matrix_mask[l] * self.adjacency_transpose_matrix[edge_type], outputs)  # [b, v, h]
            else:
                outputs = torch.matmul(self.adjacency_transpose_matrix[edge_type], outputs)

            if edge_type == 0:
                W_acts = outputs
            else:
                W_acts += outputs

        for node_type in range(len(node_type_vocab_dict)):
            U_wi = self._U_wl[node_type]
            U_bi = self._U_bl[node_type]
            outputs = torch.matmul(self.r.view(self.batch_size * self.max_seq_len, self.hidden_dim) *  hidden_inputs, U_wi) # [b*v, h]
            outputs = outputs + U_bi # [b*v, h]

            outputs = outputs.view(self.batch_size, self.max_seq_len, self.hidden_dim)  # [b, v, h]

            if node_type==0:
                U_acts = outputs * self.node_type_mask[node_type] # [b, v, h]
            else:
                U_acts += outputs * self.node_type_mask[node_type] # [b, v, h]

        h_hat = torch.tanh(W_acts + U_acts)
        return h_hat

    def reduce_final_state_with_max(self, embedding):
        # reduce max
        c_state = torch.max(embedding, dim=1)[0]
        h_state = torch.max(embedding, dim=1)[0]
        return (c_state, h_state)

    def reduce_final_state_with_root_node(self, embedding):
        # 取根节点
        c_state = embedding[:, 0, :]
        h_state = embedding[:, 0, :]
        return (c_state, h_state)


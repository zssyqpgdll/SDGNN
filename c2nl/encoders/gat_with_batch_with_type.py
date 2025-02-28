#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@project: PyCharm
@file: gat.py
@author: Shengqiang Zhang
@time: 2021/10/19 13:42
@mail: sqzhang77@gmail.com
"""

import argparse
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from c2nl.modules.util_class import LayerNorm
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.modules.embeddings import Embeddings
from c2nl.utils.misc import sequence_mask


class GraphAttentionLayer(nn.Module):

    def __init__(self, args, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat



        #self.W = torch.nn.Linear(in_features, out_features)
        #torch.nn.init.xavier_uniform_(self.W.weight)
        #torch.nn.init.constant_(self.W.bias, 0)

        self.W = nn.ModuleList([torch.nn.Linear(in_features, out_features) for i in range(args.num_node_type)])
        [torch.nn.init.xavier_uniform_(self.W[i].weight) for i in range(args.num_node_type)]
        [torch.nn.init.constant_(self.W[i].bias, 0) for i in range(args.num_node_type)]

        self.a = torch.nn.Linear(2*out_features, 1)
        torch.nn.init.xavier_uniform_(self.a.weight)
        torch.nn.init.constant_(self.a.bias, 0)


        #self.W2 = torch.nn.Linear(2*out_features, out_features)
        #torch.nn.init.xavier_uniform_(self.W2.weight)
        #torch.nn.init.constant_(self.W2.bias, 0)


        self.leakyrelu = nn.LeakyReLU(self.alpha)


        # edge type embedding matrix look-up tables
        # word_vec_size保持与out_features一致，word_vocab_size为边类型的数量+1。其中0代表是无边的，给无边连接的节点用的
        # 注意，边类型的数值从1开始。这样，遇到0就代表无边了，就会被填充了
        ########### TODO: 这里边类型有关的embedding在每个头中是独立的###########
        self.edge_type_embeddings = nn.Embedding(num_embeddings=args.num_edge_type + 1, embedding_dim=out_features, padding_idx=0)



    def forward(self, node_rep, node_type_mask_rep, adj):
        """
        Input:
            - node_rep: ``(batch_size, max_doc_len, emb_size)``
            - node_type_mask_rep: ``(batch_size, node_type_num, max_doc_len, 1)``
            - adj: ``(batch_size, max_doc_len, max_doc_len)``
        Output:
            - ``(batch_size, max_doc_len, emb_size)``
        """


        B = node_rep.size()[0] # 获取batch size大小
        N = node_rep.size()[1] # 获取节点数量N

        
        # transform for node_type
        node_type_mask_rep = node_type_mask_rep.view(self.args.num_node_type, B, N, 1) # [B, num_node_type, N, 1] -> [num_node_type, B, N, 1]
        for index in range(self.args.num_node_type):
            if index == 0:
                h = self.W[index](node_rep) * node_type_mask_rep[index] # shape [B, N, out_features]
            else:
                h = h + (self.W[index](node_rep) * node_type_mask_rep[index]) # shape [B, N, out_features]
        

        #h = self.W(node_rep) # shape [B, N, out_features]


        # h.repeat(1, N)表示第0维复制1次(等于不复制)，第1维复制N次。
        # x1[    ]        x1[    ]x1[    ]x1[    ]
        # x2[    ]   ->   x2[    ]x2[    ]x2[    ]      维度从 [N, out_features]  -> [N, N * out_features]
        # x3[    ]        x3[    ]x3[    ]x3[    ]

        # h.repeat(1, N).view(N * N, -1)表示:
        # x1[    ]x1[    ]x1[    ]        x1[    ]
        # x2[    ]x2[    ]x2[    ]  ->    x1[    ]
        # x3[    ]x3[    ]x3[    ]        x1[    ]
        #                                 x2[    ]      维度从 [N, N * out_features] -> [N * N, out_features]
        #                                 x2[    ]
        #                                 x2[    ]
        #                                 x3[    ]
        #                                 x3[    ]
        #                                 x3[    ]

        # h.repeat(N, 1)表示第0维复制N次，第1维复制1次。(等于不复制)
        # x1[    ]        x1[    ]
        # x2[    ]        x2[    ]
        # x3[    ]   ->   x3[    ]
        #                 x1[    ]
        #                 x2[    ]                      维度从 [N, out_features]  -> [N * N, out_features]
        #                 x3[    ]
        #                 x1[    ]
        #                 x2[    ]
        #                 x3[    ]

        ########### TODO: 开始构造r_(ij)矩阵了 ###########
        #
        #                       adj(B, N, N)                            edges_relation
        #                        |                                            |
        #                        V                                            V
        #                        x1  x2  x3
        #                   x1  [1] [2] [3]       adj.view(B, N * N, 1)    r(1_1) [1]    self.edge_type_embeddings(edges_relation)    r(1_1) [      ]
        # 由邻接矩阵转化而来  x2  [0] [1] [0]       --------------------->   r(1_2) [2]   ---------------------------------------->     r(1_2) [      ]
        #                   x3  [3] [2] [1]                                r(1_3) [3]                                                 r(1_3) [      ]
        #                                                                  r(2_1) [0]                                                 r(2_1) [      ]
        #                                                                  r(2_2) [1]                                                 r(2_2) [      ]
        #                                                                  r(2_3) [0]                                                 r(2_3) [      ]
        #                                                                  r(3_1) [3]                                                 r(3_1) [      ]
        #                                                                  r(3_2) [2]                                                 r(3_2) [      ]
        #                                                                  r(3_3) [1]                                                 r(3_3) [      ]

        edges_relation = self.edge_type_embeddings(adj.view(B, N * N, 1).squeeze(2))
        edges_relation = self.dropout(edges_relation)



        # h.repeat(1, N, 1) + edges_relation表示：
        # x1[    ]       r(1_1) [    ]
        # x2[    ]       r(1_2) [    ]
        # x3[    ]       r(1_3) [    ]
        # x1[    ]   +   r(2_1) [    ]
        # x2[    ]       r(2_2) [    ]
        # x3[    ]       r(2_3) [    ]
        # x1[    ]       r(3_1) [    ]
        # x2[    ]       r(3_2) [    ]
        # x3[    ]       r(3_3) [    ]


        # torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)表示:
        # x1[    ]        x1[    ]         x1[    ] x1[    ]
        # x1[    ]        x2[    ]         x1[    ] x2[    ]
        # x1[    ]        x3[    ]         x1[    ] x3[    ]
        # x2[    ]        x1[    ]         x2[    ] x1[    ]
        # x2[    ]        x2[    ]    ->   x2[    ] x2[    ]   维度从 [N * N, out_features]  -> [N*N, 2*out_features]
        # x2[    ]        x3[    ]         x2[    ] x3[    ]
        # x3[    ]        x1[    ]         x3[    ] x1[    ]
        # x3[    ]        x2[    ]         x3[    ] x2[    ]
        # x3[    ]        x3[    ]         x3[    ] x3[    ]



        # .view(B, N, -1, 2 * self.out_features)表示：
        # x1[    ] x1[    ]      x1[    ] x1[    ]
        # x1[    ] x2[    ]      x1[    ] x2[    ]
        # x1[    ] x3[    ]      x1[____] x3[____]
        # x2[    ] x1[    ]      x2[    ] x1[    ]
        # x2[    ] x2[    ]  ->  x2[    ] x2[    ]           维度从 [N*N, 2*out_features]  -> [N, N, 2*out_features]
        # x2[    ] x3[    ]      x2[____] x3[____]
        # x3[    ] x1[    ]      x3[    ] x1[    ]
        # x3[    ] x2[    ]      x3[    ] x2[    ]
        # x3[    ] x3[    ]      x3[____] x3[____]





        x_2 = (h.view(B, 1, N * self.out_features) + edges_relation.view(B, N, N * self.out_features)).view(B, N * N, self.out_features)
        #x_2 =self.W2(torch.cat([h.repeat(1, N, 1), edges_relation], dim=2))  # self.W2([B, N*N, self.out_features] || [B, N*N, self.out_features]) -> [B, N*N], self.out_features]
        x_1 = h.repeat(1, 1, N).view(B, N * N, -1)
        
        
        # concat x_1和x_2
        x_1 = torch.cat([x_1, x_2], dim=2).view(B, N, N, 2 * self.out_features)


        # torch.matmul是tensor的乘法，输入可以是高维的。输入是都是二维时，就是普通的矩阵乘法
        # a_input, [N, N, 2*out_features]      self.a, [2*out_features, 1]
        # torch.matmul(a_input, self.a)指的是：
        # [N, N, 2*out_features] x [2*out_features, 1] -> [N, N , 1]

        # a.squeeze(N) 就是去掉a中指定的维数为一的维度。
        # torch.matmul(a_input, self.a).squeeze(2)指的是：
        # [N,N,1] -> [N,N]，这里已经得到e_(ij)了
        x_1 = self.leakyrelu(self.a(x_1).squeeze(3))  # [B,N,N,1] -> [B,N,N]


        # torch.ones_like(e)指的是返回一个与e维度相同的tensor，并且其元素值都是1
        # -9e15*torch.ones_like(e)指的是将没有连接的边置为负无穷
        zero_vec = -9e15*torch.ones_like(x_1)



        # torch.where(condition,x,y)表示如果condition成立，则返回x中的元素，否则返回y中的元素
        # adj, [N,N]   e, [N, N]   zero_vec, [N, N]
        # 所以torch.where(adj > 0, e, zero_vec)表示：
        # 如果两个节点之间没有边，也就是adj[i][j]=0，那么就将对应位置的值设置为zero_vec[i][j]的值，即无穷小
        # 如果两个节点之间有边，也就是adj[i][j]>0，那么就将对应位置的值设置为e[i][j]的值，即e_(ij)
        attention = torch.where(adj > 0, x_1, zero_vec)   # [B, N, N]
        #del e
        #del zero_vec
        #gc.collect()



        # 归一化一下，得到a_{ij}
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)

        
        """
        # 这里就是更新节点特征，得到\tilde{\boldsymbol{h}}_{i}=\sigma\left(\sum_{j \in \aleph_{i}} \alpha_{i j} W \tilde{\boldsymbol{h}}_{j}\right)
        h_prime = torch.matmul(attention, h)  # [B, N, N], [B, N, out_features] --> [B, N, out_features]
        
        return F.elu(h_prime)
        """


        
        # 优化内存，复用原来的变量
        attention = attention.view(B, N, 1, N)   # [B, N, N] -> [B, N, 1, N]
        x_2 = x_2.view(B, N, N, self.out_features)  # [B, N * N, self.out_features] -> [B, N, N, self.out_features]

        x_1 = torch.matmul(attention, x_2) # [B, N, 1, N] x [B, N, N, self.out_features] = [B, N, 1, self.out_features]
        x_1 = x_1.squeeze(2) # [B, N, 1, self.out_features] -> [B, N, self.out_features]

        return F.elu(x_1)
        








class GAT(EncoderBase):
    def __init__(self, args, in_features, out_features, dropout, nheads, alpha=0.01):
        super(GAT, self).__init__()

        assert out_features % nheads == 0

        self.args = args
        self.in_features = in_features
        self.out_features = out_features

        self.attentions = nn.ModuleList([GraphAttentionLayer(args, in_features, int(out_features / nheads), dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])

        self.dropout = nn.Dropout(dropout)
        
        self.layer_norm = LayerNorm(out_features)

    def forward(self, node_rep, node_type_mask_rep, adj):
        """
        Input:
            - node_rep: ``(batch_size, max_doc_len, emb_size)``
            - node_type_mask_rep: ``(batch_size, node_type_num, max_doc_len, 1)``
            - adj: ``(batch_size, max_doc_len, max_doc_len)``
        Output:
            - ``(batch_size, max_doc_len, emb_size)``
        """

        # 这里采用concat的方式来做
        context = torch.cat([att(node_rep, node_type_mask_rep, adj) for att in self.attentions], dim=2) # [B, N, output_features]
        
        # 加残差和层归一化
        out = self.layer_norm(self.dropout(context) + node_rep) # [B, N, output_features]

        return out






if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    args = parser.parse_args()

    args.num_node_type = 15
    args.num_edge_type = 5

    batch_size = 2
    num_nodes = 3
    in_features = 4
    out_features = 4
    nheads = 4
    dropout = 0.2


    rep = torch.rand((batch_size, num_nodes, in_features)) # [Batch, Num, input_feature]
    type_mask = torch.LongTensor(np.random.randint(0, 2, [batch_size, args.num_node_type, num_nodes, 1])) # [Batch, node_type_num, max_len, 1]
    ########### TODO: 这里传入的adj的元素值是要代表不同的边的类型的 ###########
    adj = torch.LongTensor(np.random.randint(0, 4, [batch_size, num_nodes, num_nodes]))


    g = GAT(args, in_features, out_features, dropout, nheads)
    out_h = g(rep, type_mask, adj)









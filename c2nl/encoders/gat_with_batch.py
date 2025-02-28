#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@project: PyCharm
@file: gat.py
@author: Shengqiang Zhang
@time: 2021/10/19 13:42
@mail: sqzhang77@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from c2nl.modules.util_class import LayerNorm
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.utils.misc import sequence_mask


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)



    def forward(self, input, adj):

        # torch.matmul是tensor的乘法，输入可以是高维的。输入是都是二维时，就是普通的矩阵乘法.
        # 把多出的一维作为batch提出来，其他部分做矩阵乘法。
        h = torch.matmul(input, self.W) # shape [B, N, out_features]



        B = h.size()[0] # 获取batch size大小
        N = h.size()[1] # 获取节点数量N





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
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features) # shape[B, N, N, 2*out_features]


        # torch.matmul是tensor的乘法，输入可以是高维的。输入是都是二维时，就是普通的矩阵乘法
        # a_input, [N, N, 2*out_features]      self.a, [2*out_features, 1]
        # torch.matmul(a_input, self.a)指的是：
        # [N, N, 2*out_features] x [2*out_features, 1] -> [N, N , 1]

        # a.squeeze(N) 就是去掉a中指定的维数为一的维度。
        # torch.matmul(a_input, self.a).squeeze(2)指的是：
        # [N,N,1] -> [N,N]，这里已经得到e_(ij)了
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B,N,N,1] -> [B,N,N]



        # torch.ones_like(e)指的是返回一个与e维度相同的tensor，并且其元素值都是1
        # -9e15*torch.ones_like(e)指的是将没有连接的边置为负无穷
        zero_vec = -9e15*torch.ones_like(e)



        # torch.where(condition,x,y)表示如果condition成立，则返回x中的元素，否则返回y中的元素
        # adj, [N,N]   e, [N, N]   zero_vec, [N, N]
        # 所以torch.where(adj > 0, e, zero_vec)表示：
        # 如果两个节点之间没有边，也就是adj[i][j]=0，那么就将对应位置的值设置为zero_vec[i][j]的值，即无穷小
        # 如果两个节点之间有边，也就是adj[i][j]>0，那么就将对应位置的值设置为e[i][j]的值，即e_(ij)
        attention = torch.where(adj > 0, e, zero_vec)   # [B, N, N]




        # 归一化一下，得到a_{ij}
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)





        # 这里就是更新节点特征，得到\tilde{\boldsymbol{h}}_{i}=\sigma\left(\sum_{j \in \aleph_{i}} \alpha_{i j} W \tilde{\boldsymbol{h}}_{j}\right)
        h_prime = torch.matmul(attention, h)  # [B, N, N], [B, N, out_features] --> [B, N, out_features]
        # 注意，这里只是1层GAT的输出，实际是有多头的，这样的效果更好

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(EncoderBase):
    def __init__(self, in_features, out_features, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        # 全连接层，作为降维
        self.fc = nn.Linear(out_features * nheads, out_features)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training) # [B, N, input_features]

        # 这里采用concat的方式来做
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2) # [B, N, output_features x n_heads]

        # 改成只拿节点embedding的
        x = F.dropout(x, self.dropout, training=self.training)



        x = self.fc(x) # [B, N, output_features x n_heads] -> [B, N, output_features]

        return x





if __name__ == "__main__":

    batch_size = 2
    num_nodes = 3
    in_features = 4
    out_features = 5

    dropout = 0.2
    alpha = 0.01

    nheads = 4


    g = GAT(in_features, out_features, dropout, alpha, nheads)

    x = torch.rand((batch_size, num_nodes, in_features)) # [Batch, Num, input_size]
    adj = torch.ones((batch_size, num_nodes, num_nodes)) # [Batch, Num, Num]



    out_h = g(x, adj)

    print(out_h)
    print(out_h.shape)

    g.train(True)
    print(g.training)
    for a in g.attentions:
        print(a.training)


    print("\n")
    g.train(False)
    print(g.training)
    for a in g.attentions:
        print(a.training)

    print("\n")
    g.attentions[0].train(True)
    g.attentions[1].train(True)
    print(g.training)
    for a in g.attentions:
        print(a.training)






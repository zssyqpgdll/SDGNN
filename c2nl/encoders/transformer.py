"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from c2nl.modules.util_class import LayerNorm
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.utils.misc import sequence_mask
from c2nl.encoders.gat_with_batch_with_type import GAT

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 dropout,
                 max_relative_positions=0,
                 use_neg_dist=True):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(heads,
                                              d_model,
                                              d_k,
                                              d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions,
                                              use_neg_dist=use_neg_dist)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                   mask=mask, attn_type="self")
        out = self.layer_norm(self.dropout(context) + inputs)
        return self.feed_forward(out), attn_per_head


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self,
                 args,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_ff=2048,
                 d_k=64,
                 d_v=64,
                 dropout=0.2,
                 max_relative_positions=0,
                 use_neg_dist=True):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        assert num_layers % 2 == 0
        self.v_layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist)
             for i in range(num_layers // 2)])
        self.s_layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist)
             for i in range(num_layers // 2)])

        self.gat_layer = GAT(args, d_model, d_model, dropout, 8)

    def count_parameters(self):
        params = list(self.v_layer.parameters()) + list(self.s_layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, lengths=None, adjacency=None, code_node_type_mask_rep=None):
        """
        Args:
            src (`FloatTensor`): `[batch_size x src_len x model_dim]`
            lengths (`LongTensor`): length of each sequence `[batch]`
            adjacency (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        self._check_args(src, lengths)

        adjacency_multi_edges = adjacency
        out = src
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
        if adjacency is not None:
            adjacency = ~(adjacency[:, :mask.shape[2], :mask.shape[2]].bool() * ~mask)
        # Run the forward pass of every layer of the tranformer.
        representations = []
        attention_scores = []

        count = 0
        for i in range(self.num_layers // 2):
            out, attn_per_head = self.v_layer[i](out, mask)
            representations.append(out)
            attention_scores.append(attn_per_head)
            
            count = count + 1
            if count == self.num_layers // 2:
                out_ = self.gat_layer(out, code_node_type_mask_rep, adjacency_multi_edges)
                out = out + out_
                representations.append(out)

            out_, attn_per_head = self.s_layer[i](out, adjacency)
            out = out + out_
            representations.append(out)
            attention_scores.append(attn_per_head)
            
            count = count + 1
            if count == self.num_layers // 2:
                out_ = self.gat_layer(out, code_node_type_mask_rep, adjacency_multi_edges)
                out = out + out_
                representations.append(out)

        return representations, attention_scores

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphAttention_Layer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = GraphAttention_Layer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, distance):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, distance) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x, attention = self.out_att(x, distance)
        x = F.elu(x)
        return x, attention
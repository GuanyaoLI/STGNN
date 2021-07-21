import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, feat_dim, embed_dim, m_size, day):
        super(Embedding, self).__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, feat_dim)
        )
        self.weight = init.xavier_uniform_(self.weight)

    def forward(self, feat_matrix):
        ## node embedding
        embed_matrix = feat_matrix
        # 128 * (4 * 402) * (4* 402) * 402 = 128 * 402
        embed_matrix = torch.matmul(self.weight, embed_matrix.t().cpu())
        embed_matrix = F.relu(embed_matrix).cuda()
        # 402 * 128
        embed_matrix = torch.t(embed_matrix)
        return embed_matrix
    
class Encoder(nn.Module):
    def __init__(self, feature_dim, embed_dim, m_size, day):
        super(Encoder, self).__init__()
        
        from aggregators import Aggregator
        self.out_agg = Aggregator(m_size, day, feature_dim, embed_dim)
        
        from encoders import Embedding
        self.out_embed = Embedding(feature_dim, embed_dim, m_size, day)
        
        self.embed_dims = 4
        self.weight = nn.Parameter(
            # 64 * (2 * 128)
            torch.FloatTensor(self.embed_dims, embed_dim * 2)
        )
        self.weight = init.xavier_uniform_(self.weight)


    def forward(self, feat_matrix, adj_matrix):
        flow_feats = self.out_agg(feat_matrix, adj_matrix)
        self_feats = self.out_embed(feat_matrix)
        # 64 * (2 * 128) * (2 * 128) * 402 = 64 * 402
        combined = torch.cat([self_feats, flow_feats], dim = 1)
        combined = torch.matmul(self.weight,torch.t(combined).cpu())
        combined = F.relu(combined).cuda()
        combined = F.dropout(combined, p = 0.5)
        # 402 * 64
#         combined = torch.t(combined)
        return combined

# class Encoders(nn.Module):
#     def __init__(self, feature_dim, embed_dim, cnn, m_size, day):
#         super(Encoders, self).__init__()
        
#         from aggregators import MeanAggregators
#         self.out_agg = MeanAggregators(cnn, m_size, day, feature_dim, embed_dim)
        
#         self.embed_dims = 1
#         self.weight = nn.Parameter(
#             # 32 * 64
#             torch.FloatTensor(self.embed_dims, int(embed_dim/2))
#         ).cuda()
#         self.weight = init.xavier_uniform_(self.weight)
#         # if base_model != None:
#         #     self.base_model = base_model

#     def forward(self, feat_node):
#         combined = self.out_agg(feat_node)
#         # 32 * 64 * 64 * 402 = 32 * 402
#         combined = torch.matmul(self.weight,torch.t(combined))
#         combined = F.relu(combined)
#         conbined = F.dropout(combined, p = 0.1)
#         # 402 * 32
# #         combined = combined.t()
#         return combined

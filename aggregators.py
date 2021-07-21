import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from torch.nn import init

import encoders

class Aggregator(nn.Module):
    def __init__(self, m_size, day, feat_dim, embed_dim):
        super(Aggregator, self).__init__()
        from encoders import Embedding
        self.out_embed = Embedding(feat_dim, embed_dim, m_size, day)
        
    def forward(self, feat_matrix, adj_matrix):
        mask = adj_matrix        
        mask_sum= mask.sum(1, keepdim = True) + 0.00001
        mask = mask.div(mask_sum)
        
        embed_matrix = self.out_embed(feat_matrix)
        
        flow_feat = mask.cpu().mm(embed_matrix.cpu())  
        return flow_feat.cuda()

    
# class MeanAggregator(nn.Module):
#     def __init__(self, cnn, m_size, day, feature_dim, embed_dim):
#         super(MeanAggregator, self).__init__()  
#         self.cnn = cnn
#         self.out_cnn = cnn(m_size, day)
#         from encoders import Embedding
#         self.out_embed = Embedding(feature_dim, embed_dim, cnn, m_size, day)


#     def forward(self, feat_node):
#         mask = self.out_cnn(feat_node)
#         mask_sum = mask.sum(1, keepdim = True) + 0.00001
#         mask = mask.div(mask_sum)

#         embed_matrix = self.out_embed(feat_node)

#         to_feat = torch.matmul(mask, embed_matrix)
# #         to_feat = to_feat.cuda()

        
#         return to_feat
    
# class MeanAggregators(nn.Module):
#     def __init__(self, cnn, m_size, day, feature_dim, embed_dim):
#         super(MeanAggregators, self).__init__()  
#         self.cnn = cnn
#         self.out_cnn = cnn(m_size, day)
#         from encoders import Encoder
#         self.out_embed = Encoder(feature_dim, embed_dim, cnn, m_size, day)


#     def forward(self, feat_node):
#         mask = self.out_cnn(feat_node)
#         mask_sum = mask.sum(1, keepdim = True) + 0.00001
#         mask = mask.div(mask_sum)

#         embed_matrix = self.out_embed(feat_node)

#         to_feat = torch.matmul(mask, embed_matrix)
# #         to_feat = to_feat.cuda()

#         return to_feat


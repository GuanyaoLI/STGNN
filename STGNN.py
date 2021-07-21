import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from cnn import CnnDay, CnnHour
from encoders import Encoder
from Gat import GAT


class STGNN(nn.Module):
    def __init__(self, feature_dim, embed_dim, m_size, day, hour):
        super(STGNN, self).__init__()
        self.m_size = m_size
        self.graph = Encoder(feature_dim, embed_dim, m_size, hour)
        self.gat = GAT(nfeat = feature_dim, nhid = 8, nclass = 4, dropout = 0.5, nheads = 2, alpha = 0.5)
        self.cnnday = CnnDay(m_size, day)
        self.cnnhour = CnnHour(m_size, hour)
        self.embed_size = 4 + 4
        
        self.rnn = nn.GRU(self.embed_size,  self.embed_size, 1)
#         self.rnn = self.rnn.cuda()

        self.w1 = nn.Parameter(torch.FloatTensor(m_size, m_size))
        init.xavier_uniform_(self.w1)
        self.w2 = nn.Parameter(torch.FloatTensor(m_size, m_size))
        init.xavier_uniform_(self.w2)
        self.w3 = nn.Parameter(torch.FloatTensor(m_size, m_size))
        init.xavier_uniform_(self.w3)
        self.w4 = nn.Parameter(torch.FloatTensor(m_size, m_size))
        init.xavier_uniform_(self.w4)
        
        self.tran_Matrix = nn.Parameter(torch.FloatTensor(2, self.embed_size))
        init.xavier_uniform_(self.tran_Matrix)
        self.hn = nn.Parameter(torch.FloatTensor(1, self.m_size, self.embed_size))
        init.xavier_uniform_(self.hn)
        
        self.loss_fn = torch.nn.MSELoss()        


    def forward(self, feat_rent_days, feat_rent_hours, feat_return_days, feat_return_hours, distance):
        
        ####flow graph
        node_rent_days = self.cnnday(feat_rent_days)
        node_rent_hours = self.cnnhour(feat_rent_hours)
        node_return_days = self.cnnday(feat_return_days)
        node_return_hours = self.cnnhour(feat_return_hours)
        
        feat_matrix = torch.cat([node_rent_days, node_rent_hours, node_return_days, node_return_hours], dim = 1)
        
        edge_rent_days = self.cnnday(feat_rent_days)
        edge_rent_hours = self.cnnhour(feat_rent_hours)
        edge_return_days = self.cnnday(feat_return_days)
        edge_return_hours = self.cnnhour(feat_return_hours)
        
        adj_matrix = self.w1 * edge_rent_days.cpu() + self.w2 * edge_rent_hours.cpu() + self.w3 * edge_return_days.cpu() + self.w4 * edge_return_hours.cpu()
        
        out_flow = self.graph(feat_matrix, adj_matrix)
        
        
        ####distance graph
        node_rent_days_dis = self.cnnday(feat_rent_days)
        node_rent_hours_dis = self.cnnhour(feat_rent_hours)
        node_return_days_dis = self.cnnday(feat_return_days)
        node_return_hours_dis= self.cnnhour(feat_return_hours)
        
        feat_matrix_dis = torch.cat([node_rent_days_dis, node_rent_hours_dis, node_return_days_dis, node_return_hours_dis], dim = 1)
                
        out_dis, att_dis = self.gat(feat_matrix_dis, distance)
        
        
        output = torch.cat([out_flow.t(), out_dis], dim = 1).cpu()
#         output = out_dis.cpu()
        
        inputs = output.reshape(1, self.m_size, self.embed_size)
        output, hn = self.rnn(inputs, self.hn)
        self.hn = nn.Parameter(hn)
        output = output.reshape(self.m_size, self.embed_size)
        
        scores = torch.matmul(self.tran_Matrix, output.t()).cuda()
# #         scores = output  
#         att_dis = 0

        return scores, att_dis


    def loss(self, scores, ground_truth):
        ground_truth = ground_truth.cuda()
        loss = self.loss_fn(scores, ground_truth)      
        return loss

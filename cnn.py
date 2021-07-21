import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

class CnnDay(nn.Module):
    def __init__(self, m_size, day):
        super(CnnDay, self).__init__()
        self.m_size = m_size
        self.cnn = nn.Conv2d(in_channels = day,
                             out_channels = 1,
                             kernel_size = 1,
                             stride = 1,
                             padding = 0)
        self.cnn = self.cnn.cuda()

        
    def forward(self, feat_edge_days):
        feat_edge_days = torch.unsqueeze(feat_edge_days, 0)
        output = self.cnn(feat_edge_days)
        output = F.dropout(output, p = 0.5)
        output = F.relu(output)
        return output.reshape(self.m_size, self.m_size)

    
class CnnHour(nn.Module):
    def __init__(self, m_size, hour):
        super(CnnHour, self).__init__()
        self.m_size = m_size
        self.cnn = nn.Conv2d(in_channels = hour,
                     out_channels = 1,
                     kernel_size = 1,
                     stride = 1,
                     padding = 0)
        self.cnn = self.cnn.cuda()

    def forward(self, feat_edge_hours):
        feat_edge_hours = torch.unsqueeze(feat_edge_hours, 0)
        output = self.cnn(feat_edge_hours)
        output = F.dropout(output, p = 0.5)
        output = F.relu(output)
        return output.reshape(self.m_size, self.m_size)
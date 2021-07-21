import torch
import numpy as np
import pandas as pd
from scipy import sparse
        
def load_dataset(path, dataset, flow_dataset, m_size):
    df = np.load(path + dataset +'.npy')
    para = np.max(df)
    data = df / para

    sp_flow = sparse.load_npz(path + flow_dataset +'.npz')
    flow = sp_flow.toarray()
    scale = np.max(flow)
    
    flow = flow.reshape(-1, m_size, m_size)
    return data, flow, para, scale 

def load_distance(path):
    distance = np.load(path + 'distance.npy')
    distance = torch.from_numpy(distance).float().cuda()
    return distance
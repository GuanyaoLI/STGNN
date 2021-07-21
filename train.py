import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# !/bin/bash
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader

torch.cuda.empty_cache()

import numpy as np
import pandas as pd
import time

import datetime

from GetDataset import BikeDataset
from utils import load_distance, load_dataset
from STGNN import STGNN

def train(epoch):
    model.train()

    loss_train_one = torch.zeros(1).cuda()    

    for i in range(train_day * hour):  
        print(day_idxs[i], hour_idxs[i], gt_idxs[i])
        feat_rent_days = torch.from_numpy(rent_flow[day_idxs[i][-1],:,:]).float().cuda()
        feat_return_days = torch.from_numpy(return_flow[day_idxs[i][-1],:,:]).float().cuda()
        feat_rent_hours = torch.from_numpy(rent_flow[hour_idxs[i][-1],:,:]).float().cuda()       
        feat_return_hours = torch.from_numpy(return_flow[hour_idxs[i][-1],:,:]).float().cuda()
        
        ground_truth_rent = rent_data[gt_idxs[i]].reshape(-1,m_size)
        ground_truth_return = return_data[gt_idxs[i]].reshape(-1,m_size)
        ground_truth = torch.from_numpy(np.concatenate((ground_truth_rent, ground_truth_return),axis = 0)).float()        
        scores, att = model(feat_rent_days, feat_rent_hours, feat_return_days, feat_return_hours, distance)
        loss_train_one += model.loss(scores, ground_truth)

        if (i + 1) % batch_size == 0:
            loss_train = loss_train_one.div(batch_size)
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train_one = torch.zeros(1).cuda()
            print('loss_train = ', loss_train.item())
    
    model.eval()
    loss_val_one = torch.zeros(1).cuda()
    for j in range(train_day * hour, (train_day + vali_day) * hour): 
        with torch.no_grad():
            print(day_idxs[j], hour_idxs[j], gt_idxs[j])
            feat_rent_days = torch.from_numpy(rent_flow[day_idxs[j][-1],:,:]).float().cuda()
            feat_return_days = torch.from_numpy(return_flow[day_idxs[j][-1],:,:]).float().cuda()
            feat_rent_hours = torch.from_numpy(rent_flow[hour_idxs[j][-1],:,:]).float().cuda()       
            feat_return_hours = torch.from_numpy(return_flow[hour_idxs[j][-1],:,:]).float().cuda()
            
            ground_truth_rent = rent_data[gt_idxs[j]].reshape(-1,m_size)
            ground_truth_return = return_data[gt_idxs[j]].reshape(-1,m_size)
            ground_truth = torch.from_numpy(np.concatenate((ground_truth_rent, ground_truth_return),axis = 0)).float()        
            scores, att = model(feat_rent_days, feat_rent_hours, feat_return_days, feat_return_hours, distance)

            loss_val_one += model.loss(scores, ground_truth)

    loss_val = loss_val_one.div(vali_day * hour)
    print('-----------------------------------')
    print('loss_val = ', loss_val.item())            
    return loss_val.item()

def test():
    model.eval()
    result = []
    ground = [] 
    att_days = []
    att_hours = []
    
    for i in range((train_day + vali_day)*hour, (train_day + vali_day + test_day) * hour):  
        print(i)
        feat_rent_days = torch.from_numpy(rent_flow[day_idxs[i][-1],:,:]).float().cuda()
        feat_return_days = torch.from_numpy(return_flow[day_idxs[i][-1],:,:]).float().cuda()
        feat_rent_hours = torch.from_numpy(rent_flow[hour_idxs[i][-1],:,:]).float().cuda()       
        feat_return_hours = torch.from_numpy(return_flow[hour_idxs[i][-1],:,:]).float().cuda()
        
        ground_truth_rent = rent_data[gt_idxs[i]].reshape(-1,m_size)
        ground_truth_return = return_data[gt_idxs[i]].reshape(-1,m_size)
        ground_truth = np.concatenate((ground_truth_rent, ground_truth_return),axis = 0)     
        pre, att = model(feat_rent_days, feat_rent_hours, feat_return_days, feat_return_hours, distance)
        
        result.append(pre.detach().cpu().numpy())
        ground.append(ground_truth)    
        
    result = np.array(result)
    ground = np.array(ground)
    
    return result, ground

start_time = time.time()


path= '/data/la/'


rent_dataset = 'pickup'
rent_flow_dataset = 'sparse_rent2return'

return_dataset = 'dropoff'
return_flow_dataset = 'sparse_return2rent'

#all day
day = 7
hour = 96
m_size = 135

train_day = 60
vali_day = 30
test_day = 450

# # weekday
# day = 5
# hour = 96
# m_size = 135

# train_day = 43
# vali_day = 21
# test_day = 320

# # weekend
# day = 2
# hour = 96
# m_size = 135

# train_day = 17
# vali_day = 9
# test_day = 129


learning_rate = 0.01 
epoch_no = 100
batch_size = 32
embed_dim = 8
feature_dim = m_size * 4

rent_data, rent_flow, rent_para, scale = load_dataset(path, rent_dataset, rent_flow_dataset, m_size)
return_data, return_flow, return_para, scale = load_dataset(path, return_dataset, return_flow_dataset, m_size)


distance = load_distance(path)
day_idxs = []
hour_idxs = []
gt_idxs = []

for n in range(day* hour, (day + train_day + vali_day + test_day) * hour):
    day_idxs.append([(n - i) for i in range((day) * hour, 0, -hour)])
    hour_idxs.append(list(range((n - hour), n)))
    gt_idxs.append(n)
print(len(gt_idxs))

      
model = STEMGEM(feature_dim, embed_dim, m_size, day, hour)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma=0.1) 


loss_values = []
bad_counter = 0
best = epoch_no + 1
best_epoch = 0


for epoch in range(epoch_no):
    print('Training Process: epoch = ', epoch)
    loss_values.append(train(epoch))    
    scheduler.step()
    torch.save(model.state_dict(), path + '/Result/' +'{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == 20:
        break

    files = glob.glob(path + '/Result/' + '*.pkl')
    for file in files:
        epoch_nb = int((file.split('.')[0]).split('/')[-1])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob(path + '/Result/' + '*.pkl')
for file in files:
    epoch_nb = int((file.split('.')[0]).split('/')[-1])
    if epoch_nb > best_epoch:
        os.remove(file)
#     Save Parameters

#     torch.save(model.state_dict(), path + station + '/' + station + '.pkl') 
with torch.no_grad():
    predict, ground_truth = test()
        
df1 = predict
np.save(path + 'Result/'+ 'pr.npy', df1)
df2 = ground_truth
np.save(path + 'Result/' + 'gt.npy', df2)
torch.cuda.empty_cache()



end_time = time.time()
times = end_time - start_time
print('Total_time =',times)
torch.cuda.empty_cache()
print('max = ', para)




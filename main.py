#!/usr/bin/env python
# coding: utf-8

# In[21]:


import models
import torch
import torch.nn as nn
import pandas as pd
from data import Data_Utility
from utils.visualize import Visualizer
from tqdm import tqdm
import joblib
from torchnet import meter
from config import opt
from loss import sigmoid_mse_loss


# In[111]:


DAY_POINTS = 48


# In[129]:


@torch.no_grad()
def val(model,data_utility,test_X,test_Y,device):
    model.eval()
    
    total_loss = 0
    total_pred_loss = 0
    
    for i,(data,label) in tqdm(enumerate(data_utility.get_batches(test_X,test_Y,opt.batch_size))):
        inputs = data.to(device)
        targets = label.to(device)
        
        preds = model(inputs)
        preds = preds.squeeze(2)
        
        loss = torch.mean(torch.pow(targets-preds,2))
        total_loss += loss
        
        pred_loss = torch.mean(torch.pow(targets[:,-opt.P:]-preds[:,-opt.P:],2))
        total_pred_loss += pred_loss
    
    model.train()
    
    return total_loss/(i+1),total_pred_loss/(i+1)


# In[88]:


def load_test_X_Y(data,data_utility):
    data = data.copy()
    data.drop(columns=['date'],inplace=True)
    
    scaler = data_utility.get_scaler()
    lbes = data_utility.get_lbes()
    
    data[opt.dense_features] = scaler.transform(data[opt.dense_features])
    for feat in opt.sparse_features:
        data[feat] = lbes[feat].transform(data[feat])
    y_data = data[opt.target].values
    
    data = data_utility.build_dataset(data)
    X,Y = data_utility.build_X_Y(data,y_data,opt.P,opt.WINDOW_SIZE)
    return X,Y


# In[36]:


def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port=opt.vis_port)
    
    train_data = pd.read_csv('./raw_data/train_data.csv')
    train_data.drop(columns=['date'],inplace=True)
    
    data_utility = Data_Utility(train_data,opt.sparse_features,opt.dense_features,
                            opt.target,opt.WINDOW_SIZE,opt.embedding_dim,opt.P)
    joblib.dump(data_utility,'data_utility.pkl')
    
    feature_columns = data_utility.get_feature_columns()
    feature_index = data_utility.get_feature_index()
    
    model = models.DTCN(feature_columns=feature_columns,feature_index=feature_index,
                    dnn_hidden_units=opt.dnn_hidden_units,window_size=opt.WINDOW_SIZE,
                    output_size=opt.out_size,residual_size=opt.residual_size,
                    skip_size=opt.skip_size,dilation_cycles=opt.dilation_cycles,
                    dilation_depth=opt.dilation_depth)
    
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    model.to(device)
    
    train_X,train_Y = data_utility.get_data()
    test_data = pd.read_csv('./raw_data/test_data.csv')
    test_X,test_Y = load_test_X_Y(test_data,data_utility)
    
#     criterion = sigmoid_mse_loss(window_size=opt.WINDOW_SIZE,device=device)
    criterion = nn.MSELoss()
#     criterion.to(device)
    
    lr = opt.lr
    optimizer = model.get_optimizer(lr,opt.weight_decay)
    loss_meter = meter.AverageValueMeter()
    
    previous_loss = 1e10
    
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        for i,(data,label) in tqdm(enumerate(data_utility.get_batches(train_X,train_Y,opt.batch_size))):
            inputs = data.to(device)
            targets = label.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            preds = preds.squeeze(2)
            loss = criterion(preds,targets)
            
            loss.backward()
            optimizer.step()
            
            loss_meter.add(loss.item())
            if (i+1)%opt.print_freq == 0:
                vis.plot('loss',loss_meter.value()[0])
        
        val_loss,val_pred_loss = val(model,data_utility,test_X,test_Y,device)
        vis.plot('val_loss',val_loss.item())
        vis.plot('val_pred_loss',val_pred_loss.item())
        
        save_name = 'models/checkpoints/'+opt.model+str(epoch)+'.pth'
        model.save(save_name)
        
        if loss_meter.value()[0]>previous_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        previous_loss = loss_meter.value()[0]


if __name__=='__main__':
    import fire
    fire.Fire()
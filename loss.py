#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


class sigmoid_mse_loss(nn.Module):
    def __init__(self,window_size,device):
        super(sigmoid_mse_loss,self).__init__()
        self.idx = torch.tensor([i/window_size*10-5 for i in range(1,window_size+1)])
        self.weight = 2*torch.sigmoid(self.idx)
        self.weight = self.weight.to(device)
        
    def forward(self,y_pred,y_true):
        return torch.mean(torch.pow(y_true-y_pred,2)*self.weight)


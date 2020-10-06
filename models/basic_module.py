#!/usr/bin/env python
# coding: utf-8

# In[59]:


import torch
import torch.nn as nn
import time


# In[60]:


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))
        
    def load(self,path):
        self.load_state_dict(torch.load(path))
        
    def save(self,name=None):
        if name is None:
            prefix = 'checkpoints/'+self.model_name+'_'
            name = time.strftime(prefix+'%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(),name)
        return name
    
    def get_optimizer(self,lr,weight_decay):
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)


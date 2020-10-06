#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torch.nn as nn

# In[2]:


class CausalConv1d(nn.Module):
    def __init__(self,in_size,out_size,kernel_size,dilation=1):
        super(CausalConv1d,self).__init__()
        self.pad = (kernel_size - 1)*dilation
        self.conv1 = nn.Conv1d(in_size,out_size,kernel_size,padding=self.pad,dilation=dilation)
    
    def forward(self,x):
        x = self.conv1(x)
        x = x[...,:-self.pad]
        return x


# In[9]:


class ResidualLayer(nn.Module):
    def __init__(self,residual_size,skip_size,dilation):
        super(ResidualLayer,self).__init__()
        self.conv_filter = CausalConv1d(residual_size,residual_size,kernel_size=2,dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size,residual_size,kernel_size=2,dilation=dilation)
        
        self.resconv1_1 = nn.Conv1d(residual_size,residual_size,kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size,skip_size,kernel_size=1)
        
    def forward(self,x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)
        fx = torch.tanh(conv_filter)*torch.sigmoid(conv_gate)
        fx = self.resconv1_1(fx)
        skip = self.skipconv1_1(fx)
        residual = fx + x
        return skip,residual


# In[41]:


class DilatedStack(nn.Module):
    def __init__(self,residual_size,skip_size,dilation_depth):
        super(DilatedStack,self).__init__()
        residual_stack = [ResidualLayer(residual_size,skip_size,2**layer) for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)
        
    def forward(self,x):
        skips = []
        for layer in self.residual_stack:
            skip,x = layer(x)
            skips.append(skip.unsqueeze(0))
        return torch.cat(skips,dim=0),x


# In[51]:


class WaveNet(nn.Module):

    def __init__(self,input_size,out_size, residual_size, skip_size, dilation_cycles, dilation_depth):

        super(WaveNet, self).__init__()
        
        self.model_name = 'wavenet'

        self.input_conv = CausalConv1d(input_size,residual_size, kernel_size=2)        

        self.dilated_stacks = nn.ModuleList(

            [DilatedStack(residual_size, skip_size, dilation_depth)

             for cycle in range(dilation_cycles)]

        )

        self.convout_1 = nn.Conv1d(skip_size, out_size, kernel_size=1)

        self.convout_2 = nn.Conv1d(out_size, out_size, kernel_size=1)

    def forward(self, x):

        x = x.permute(0,2,1)# [batch,input_feature_dim, seq_len]

        x = self.input_conv(x) # [batch,residual_size, seq_len]             

        skip_connections = []

        for cycle in self.dilated_stacks:

            skips, x = cycle(x)             
            skip_connections.append(skips)

        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)        

        # gather all output skip connections to generate output, discard last residual output

        out = skip_connections.sum(dim=0) # [batch,skip_size,seq_len]

        out = torch.relu(out)

        out = self.convout_1(out) # [batch,out_size,seq_len]
        out = torch.relu(out)

        out=self.convout_2(out)

        out=out.permute(0,2,1)
        #[bacth,seq_len,out_size]
        return out     


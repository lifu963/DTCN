#!/usr/bin/env python
# coding: utf-8

# In[3]:
import sys
sys.path.append('../')

import torch
import torch.nn as nn
from inputs import SparseFeat,DenseFeat
import numpy as np


# In[4]:


class dnn(nn.Module):
    '''
    input_size:(B,f)
    output_size:(B,hidden_units[-1])
    '''
    def __init__(self,inputs_dim,hidden_units,dropout_rate=0,use_bn=False,
                 init_std=0.0001,seed=1024):
        super(dnn, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i],hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
            [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [nn.ReLU(inplace=True) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)


    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


# In[5]:


class CrossNet(nn.Module):
    def __init__(self,in_features,layer_num=2,seed=1024):
        super(CrossNet,self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features,1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features,1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


# In[6]:


class dcn(nn.Module):
    '''
    input_size:(B,f)
    output_size:(B,hidden_units[-1]+f)
    '''
    def __init__(self,input_dim,dnn_hidden_units,dnn_use_bn,dnn_dropout,dnn_init_std,
                 cross_num=2,seed=1024):
        super(dcn,self).__init__()
        self.dnn = dnn(input_dim,dnn_hidden_units,use_bn=dnn_use_bn,
                       dropout_rate=dnn_dropout,init_std=dnn_init_std)
        self.crossnet = CrossNet(input_dim,layer_num=cross_num,seed=seed)
        
    def forward(self,X):
        deep_out = self.dnn(X)
        cross_out = self.crossnet(X)
        stack_out = torch.cat((cross_out,deep_out),dim=-1)
        return stack_out


# In[7]:


class DCN(nn.Module):
    '''
    input_size:(B,W,f)
    (B,W,f)->(B*W,f)->(B*W,f1)->(B*W,hidden_units[-1]+f1)->(B,W,hidden_units[-1]+f1)
    f2 = hidden_units[-1]+f1
    output_size:(B,W,f2)
    '''
    def __init__(self,dnn_feature_columns,feature_index,dnn_hidden_units,dnn_use_bn=False,dnn_dropout=0,dnn_init_std=0.0001,
                 embed_init_std=0.0001,cross_num=2,seed=1024,l2_reg_embedding=1e-5,l2_reg_cross=1e-5,l2_reg_dnn=0):
        super(DCN,self).__init__()
        self.feature_columns = dnn_feature_columns
        self.input_dims = self.compute_input_dim(self.feature_columns)
        self.feature_index = feature_index
        self.dcn = dcn(self.input_dims,dnn_hidden_units,dnn_use_bn,dnn_dropout,dnn_init_std,
                 cross_num,seed)
        self.embedding_dict = self.create_embedding_matrix(self.feature_columns,embed_init_std)
        
        self.reg_loss = torch.zeros((1,))
        
        self.add_regularization_loss(self.embedding_dict.parameters(),l2_reg_embedding)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0],self.dcn.dnn.named_parameters()),l2_reg_dnn)
        self.add_regularization_loss(self.dcn.crossnet.kernels,l2_reg_cross)
        
        
    def forward(self,X):
        X = X.reshape(-1,X.shape[2])
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:,self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) 
                                 for feat in self.sparse_feature_columns]
        dense_value_list = [X[:,self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] 
                            for feat in self.dense_feature_columns]    
        dnn_input = self.combined_dnn_input(sparse_embedding_list, dense_value_list)
        output = self.dcn(dnn_input)
        return output
    
    def compute_input_dim(self,feature_columns,include_sparse=True,include_dense=True):
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x,SparseFeat),feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x,DenseFeat),feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(map(lambda x: x.dimension,self.dense_feature_columns))
        sparse_input_dim = sum(feat.embedding_dim for feat in self.sparse_feature_columns)
        
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim
    
    def concat_fun(self,inputs,axis=-1):
        if len(inputs) == 1:
            return inputs[0]
        else:
            return torch.cat(inputs,dim=axis)
    
    def combined_dnn_input(self,sparse_embedding_list,dense_value_list):
        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list,dim=-1),start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list,dim=-1),start_dim=1)
        return self.concat_fun([sparse_dnn_input.float(),dense_dnn_input.float()])
    
    def create_embedding_matrix(self,feature_columns,init_std=0.0001,linear=False,sparse=False,device='cpu'):
        embedding_dict = nn.ModuleDict(
        {feat.embedding_name:nn.Embedding(feat.vocabulary_size,feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in self.sparse_feature_columns})
        
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
            
        return embedding_dict
    
    def add_regularization_loss(self,weight_list,weight_decay,p=2):
        reg_loss = torch.zeros((1,))
        for w in weight_list:
            if isinstance(w,tuple):
                l2_reg = torch.norm(w[1],p=p,)
            else:
                l2_reg = torch.norm(w,p=p,)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        self.reg_loss = self.reg_loss + reg_loss


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler,LabelEncoder
from inputs import SparseFeat,DenseFeat
from collections import OrderedDict
import numpy as np


# In[2]:


class Data_Utility(object):
    def __init__(self,data,sparse_features,dense_features,target,window_size,embedding_dim,P=1):
        
        self.window_size = window_size
        self.P = P        
#       连续性特征归一化
        self.scaler = StandardScaler()
        data[dense_features] = self.scaler.fit_transform(data[dense_features])
#       类别型特征编码
        self.lbes = OrderedDict()
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
            self.lbes[feat] = lbe
            del lbe
        y_data = data[target].values
            
        self.embedding_dim = embedding_dim
        self.dnn_feature_columns = [SparseFeat(feat,vocabulary_size=data[feat].nunique(),embedding_dim=self.embedding_dim[feat]) 
        for feat in sparse_features] + [DenseFeat(feat,1,) for feat in dense_features]
        
        self.feature_index = self.build_input_features(self.dnn_feature_columns)
        self.feature_names = list(self.feature_index.keys())
        
        self.train_set = self.build_dataset(data)
        self.y_data = y_data
        
        X,Y = self.build_X_Y(self.train_set,self.y_data,self.P,self.window_size)
        
        self.X = X
        self.Y = Y
    
    def build_X_Y(self,data_set,y_data,P,window_size):
        set_size = (data_set.shape[0]-P)//window_size
        data_set = data_set[:set_size*window_size]
        y_data = y_data[P:set_size*window_size+P]
        
        train_batchs = range(0,len(data_set)-window_size+1)
        batch_sum = len(train_batchs)
        n_features = data_set.shape[1]
        
        X = torch.zeros((batch_sum,window_size,n_features))
        Y = torch.zeros((batch_sum,window_size))
        
        for i in range(batch_sum):
            start = train_batchs[i]
            end = train_batchs[i]+window_size
            X[i,:,:]=torch.from_numpy(data_set[start:end,:])
            Y[i,:]=torch.from_numpy(y_data[start:end])
        
        return X,Y
        
        
    def get_data(self):
        return [self.X,self.Y]
    
    def get_feature_columns(self):
        return self.dnn_feature_columns
    
    def get_feature_index(self):
        return self.feature_index
    
    def get_feature_names(self):
        return self.feature_names
    
    def get_scaler(self):
        return self.scaler
    
    def get_lbes(self):
        return self.lbes
    
    def build_input_features(self,feature_columns):
        features = OrderedDict()
        start = 0
        for feat in feature_columns:
            feat_name = feat.name
            if feat_name in features:
                continue
            if isinstance(feat, SparseFeat):
                features[feat_name] = (start, start + 1)
                start += 1
            elif isinstance(feat, DenseFeat):
                features[feat_name] = (start, start + feat.dimension)
                start += feat.dimension
            else:
                raise TypeError("Invalid feature column type,got", type(feat))
        return features
        
    def build_dataset(self,data):
        data_model_input = {name:data[name] for name in self.feature_names}        
        data_set = [data_model_input[feature] for feature in self.feature_index]
        for i in range(len(data_set)):
            if len(data_set[i].shape)==1:
                data_set[i] = np.expand_dims(data_set[i],axis=1)
        data_set = np.concatenate(data_set,axis=-1)
        return data_set
    
    def get_batches(self,inputs,targets,batch_size,shuffle=True):
        inputs_len = len(inputs)
        if shuffle:
            index = torch.randperm(inputs_len)
        else:
            index = torch.LongTensor(range(inputs_len))
            
        start_idx = 0
        while(start_idx<inputs_len):
            end_idx = min(inputs_len,start_idx+batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            yield Variable(X),Variable(Y)
            start_idx += batch_size


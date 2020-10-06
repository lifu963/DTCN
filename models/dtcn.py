#!/usr/bin/env python
# coding: utf-8

# In[10]:


from .basic_module import BasicModule
from .wavenet import WaveNet
from .DCN import DCN


# In[12]:


class DTCN(BasicModule):
    def __init__(self,feature_columns,feature_index,dnn_hidden_units,window_size,
                 output_size,residual_size,skip_size,dilation_cycles,dilation_depth,
                 dnn_use_bn=False,dnn_dropout=0,dnn_init_std=0.0001,embed_init_std=0.0001,cross_num=2,seed=1024,
                 l2_reg_embedding=1e-05,l2_reg_cross=1e-05,l2_reg_dnn=0):
        super(DTCN,self).__init__()
        self.DCN = DCN(feature_columns,feature_index,dnn_hidden_units,dnn_use_bn,dnn_dropout,dnn_init_std,
    embed_init_std,cross_num,seed,l2_reg_embedding,l2_reg_cross,l2_reg_dnn)
        self.dcn_out_size = self.DCN.input_dims+dnn_hidden_units[-1]
        self.WaveNet = WaveNet(self.dcn_out_size,output_size,residual_size,skip_size,dilation_cycles,dilation_depth)
        self.window_size = window_size
        
    def forward(self,x):
        stack_out = self.DCN(x)
        stack_out = stack_out.reshape(-1,self.window_size,self.dcn_out_size)
        output = self.WaveNet(stack_out)
        return output


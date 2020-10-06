# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    env = 'dtcn'  # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'DTCN'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    train_data_root = './raw_data/train_data.csv'
    data_utility = None
    
    WINDOW_SIZE = 144
    P = 12
    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    
    sparse_features = ['month','day','weekday','hour']
    embedding_dim = {'month':3,'day':4,'weekday':3,'hour':4}
    dense_features = ['max_T','min_T','avg_T','D','Power']
    target = 'Power'
    
    input_size = 9
    out_size = 1
    residual_size = 5
    skip_size = 5
    dilation_cycles = 2
    dilation_depth = 4
    
    dnn_hidden_units = [10,5]
    cross_num= 2

    max_epoch = 50
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5  # 损失函数


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
#         opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset,Dataset
import numpy as np
from colorama import Fore
from matplotlib import pyplot as plt
import math
import time
from Models.Gated_Transformer import Gated_Transformer


# #### 常数设置

# In[2]:


class Const(object):
    def __init__(self):
        
        # 超参数设置
        self.epoch=50
        self.lr=1e-5
        self.device='cuda'
        self.batchsize=128
        
        # 模式选择  
        self.nwp= True # True of False
        self.type= 'long' # short : 24->4  # long: 96->96 # Custom: 自定义
        
        # 相关定义
        if self.nwp == False:
            self.f_in=1
        else:
            self.f_in=3
                   
        if self.type=='short':
            self.seq_len=24
            self.pred_len=4
            self.label_len=6
        elif self.type=='long':
            self.seq_len=96
            self.pred_len=96
            self.label_len=24           

Co=Const()


# #### 数据读取

# In[3]:


class Dataset_customer(Dataset):
    def __init__(self,df,size):
        # 去除时间列, 修改np 类型 object-->float64
        self.data=df.values[:,1:].astype('float64') 
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

    def __getitem__(self, index):
        # 选取 过去'POWER 作为X1
        X1 = self.data[index:index+self.seq_len,0:1]
        # 选取 过去'SPEED' 作为X2
        X2 = self.data[index:index+self.seq_len,1:2]
        # 选取 未来'SPEED' 作为 X3
        X3 = self.data[index+self.seq_len:index+self.seq_len+self.pred_len,1:2]
        # 选取 POWER 作为 y
        y = self.data[index+self.seq_len:index+self.seq_len+self.pred_len,0:1]
        return X1,X2,X3,y
    def __len__(self):    
        return len(self.data) - self.seq_len - self.pred_len + 1
# Read and split    
df=torch.load('Data/JSFD02/JSFD02')
LEN=len(df); 
train_len=int(LEN*(0.7))
valid_len=int(LEN*(0.2))
test_len=int(LEN*(0.1))
df_train=df.iloc[0:train_len]
df_valid=df.iloc[train_len:train_len+valid_len]
df_test=df.iloc[-test_len:]
size=[Co.seq_len,Co.label_len,Co.pred_len]
# Dataset
Dataset_train=Dataset_customer(df_train,size)
Dataset_valid=Dataset_customer(df_valid,size)
Dataset_test=Dataset_customer(df_test,size)
Loader_train=DataLoader(Dataset_train,batch_size=Co.batchsize,
                        drop_last=True, shuffle=True)
Loader_valid=DataLoader(Dataset_valid,batch_size=Co.batchsize,
                        drop_last=True, shuffle=True)
Loader_test=DataLoader(Dataset_test,batch_size=Co.batchsize,
                       drop_last=True, shuffle=True)
# [0]: 过去 POWER+SPEED [1]: 未来 SPEED  [2]:未来 POWER


# #### 模型设置

# In[4]:


class Model(nn.Module):
    ''' GatedTransformer:
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.Former=Gated_Transformer(num_encoder_layers=1,num_decoder_layers=1,gate=True)
        if Co.type=='short':
            self.Enc_embed=nn.Conv1d(Co.f_in,128,kernel_size=3,padding=1,dilation=1)
        elif Co.type=='long':
            self.Enc_embed=net=nn.Sequential(
                    nn.Conv1d(Co.f_in,128,kernel_size=3,padding=1,dilation=1),
                    nn.Conv1d(128,128,kernel_size=3,padding=2,dilation=2),
                    nn.Conv1d(128,128,kernel_size=3,padding=4,dilation=4),
                    )
        self.Dec_embed=nn.Conv1d(1,128,kernel_size=3,padding=1)
        self.Enc_pos=PositionalEmbedding()
        self.Dec_pos=PositionalEmbedding()  
        self.FC1=nn.Linear(128,1)
        self.FC2=nn.Linear(Co.seq_len+Co.label_len,Co.pred_len)  
        self.Relu=nn.ReLU()        
      
    def forward(self,X_enc,X_dec): # [B,X_L,X_F] in, [B,y_L,y_F] out.
        X1=self.Enc_embed(X_enc.transpose(1,2)).transpose(1,2)
        X2=self.Dec_embed(X_dec.transpose(1,2)).transpose(1,2)
        X_enc=X1+self.Enc_pos(X_enc)        
        X_dec=X2+self.Dec_pos(X_dec)
        out,_=self.Former(X_enc,X_dec) #[B,L,F]-->[B,L,C] 只和X_dec有关
        out = self.FC1(out).squeeze(-1)
        out = self.FC2(out).unsqueeze(-1)
        attns=None
        return out,attns
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=128, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] 


# #### 训练和验证

# In[5]:


def XyProcess(X1,X2,X3,y):
    # Process
    X1=X1.float().to(Co.device); 
    X2=X2.float().to(Co.device); 
    X3=X3.float().to(Co.device)
    y=y.float().to(Co.device)
    # Reconstruct
    if Co.nwp == True:
        X3=torch.cat([X2,X3],dim=1)[:,-Co.seq_len:,:]
        X=torch.cat([X1,X2,X3],dim=-1)
    else:
        X=X1
    X_enc=X
    X_dec1 = X1[:,-Co.label_len:,:]
    X_dec2 = torch.zeros_like(X1).float().to(Co.device)
    X_dec = torch.cat([X_dec1,X_dec2],dim=1)
    return X_enc,X_dec,y


# In[6]:


def Valid():
    global best_valid_loss
    model.eval()
    LOSS=[]
    for X1,X2,X3,y in Loader_valid:
        #--------------
        X_enc,X_dec,y=XyProcess(X1,X2,X3,y)
        #--------------
        out,attns=model(X_enc,X_dec)
        loss=crt(out,y)
        #-------------
        LOSS.append(loss.item())
    LOSS=np.mean(LOSS)
    t2=time.time()
    print(f'valid_loss={LOSS:.4f},time={t2-t1:.2f}s/epoch')
    if LOSS<best_valid_loss:
        best_valid_loss=LOSS
        print(Fore.GREEN+f'best_valid_loss={best_valid_loss:.4f}'+Fore.RESET)
        torch.save(model,path_model)
        metric['valid_loss'][ITER]=LOSS
    model.train()


# In[7]:


def Test():
    model=torch.load(path_model)
    model.eval()
    crt1=nn.MSELoss()
    crt2=nn.L1Loss() # MAE
    LOSS1=[]; LOSS2=[];
    for X1,X2,X3,y in Loader_test:
        #--------------
        X_enc,X_dec,y=XyProcess(X1,X2,X3,y)
        #--------------
        out,attns=model(X_enc,X_dec)
        loss1=crt1(out,y)
        loss2=crt2(out,y)
        #-------------
        LOSS1.append(loss1.item())
        LOSS2.append(loss2.item())
    LOSS1=np.mean(LOSS1)
    LOSS2=np.mean(LOSS2)
    print(Fore.RED+f'test_MSEloss={LOSS1:.4f}'+Fore.RESET) 
    metric['test_loss']['MSE'][ITER]=LOSS1
    metric['test_loss']['MAE'][ITER]=LOSS2


# In[8]:


metric={'valid_loss':torch.empty(5,1),
        'test_loss':  {'MSE':torch.empty(5,1),
                       'MAE':torch.empty(5,1) }}


# In[9]:


for ITER in range(0): #大循环5次，取平均值
    print('='*20+f'  ITER = {ITER}  '+'='*20)
    path_model=f'Checkpt/G_Former/G_Former_[NWP={Co.nwp}][Type={Co.type}][Iter={ITER}].pt'
    path_metric=f'Checkpt/G_Former/metric[NWP={Co.nwp}][Type={Co.type}].pt'
    model=Model().to(Co.device)
    opt=torch.optim.Adam(model.parameters(),lr=Co.lr)
    crt=nn.MSELoss()
    best_valid_loss=float('inf')
    for epoch in range(50):
        LOSS=[]
        t1=time.time()
        for X1,X2,X3,y in Loader_train:
            #--------------
            X_enc,X_dec,y=XyProcess(X1,X2,X3,y)
            #--------------
            opt.zero_grad()
            out,attns=model(X_enc,X_dec)
            loss=crt(out,y)
            loss.backward()
            opt.step()
            #-------------
            LOSS.append(loss.item())
        LOSS=np.mean(LOSS)
        print(f'epoch={epoch},train_loss={LOSS:.4f},',end='')
        Valid()
    Test()
torch.save(metric,path_metric)


# #### Metric输出

# In[10]:


#
print(f'NWP is [{Co.nwp}], Type is [{Co.type}]')


# In[11]:


#
MSE_mean=torch.mean(metric['test_loss']['MSE'])
MSE_1=max(metric['test_loss']['MSE'])-MSE_mean
MSE_2=MSE_mean-min(metric['test_loss']['MSE'])
print(f'MSE平均值为{MSE_mean:.4f},误差为+{MSE_1.item():.4f},-{MSE_2.item():.4f}')


# In[12]:


#
MAE_mean=torch.mean(metric['test_loss']['MAE'])
MAE_1=max(metric['test_loss']['MAE'])-MAE_mean
MAE_2=MAE_mean-min(metric['test_loss']['MAE'])
print(f'MAE平均值为{MAE_mean:.4f},误差为+{MAE_1.item():.4f},-{MAE_2.item():.4f}')


# In[ ]:


#----------------------[ 文件转换 ]------------------------------
# 需要保存后再操作,因为无法实时保存。
#!jupyter nbconvert --to script Github_Main.ipynb


# In[ ]:





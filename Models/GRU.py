import torch.nn as nn
import torch

class Model(nn.Module):
    ''' nn.GRU
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.gru1=nn.GRU(Co.f_in,128,batch_first=True)
        self.gru2=nn.GRU(128,128,batch_first=True)
        self.FC1=nn.Linear(128,1)
        self.FC2=nn.Linear(Co.seq_len,Co.pred_len)
        self.drop=nn.Dropout(0.2)

    def forward(self,X): 
        out,hn = self.gru1(X)
        out,hn = self.gru2(self.drop(out))
        out = self.FC1(out).squeeze(-1) 
        out = self.FC2(out).unsqueeze(-1)
        return out   

mport torch.nn as nn
import torch

class Model(nn.Module):
    ''' nn.DNN:
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.lay1=nn.Linear(Co.f_in,128)
        self.lay2=nn.Linear(128,128)
        self.FC1=nn.Linear(128,1)
        self.FC2=nn.Linear(Co.seq_len,Co.pred_len)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.2)
        

    def forward(self,X): # [B,L,F] and [B,L,F] in
        out = self.dropout(self.relu(self.lay1(X)))
        out = self.dropout(self.relu(self.lay2(out)))
        out = self.FC1(out).squeeze(-1) 
        out = self.FC2(out).unsqueeze(-1)
        return out

import torch
import torch.nn as nn
class Model(nn.Module):
    ''' nn.TCN:
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=nn.Conv1d(
            Co.f_in,128,kernel_size=3,padding=1,dilation=1)
        self.conv2=nn.Conv1d(
            128,128,kernel_size=3,padding=2,dilation=2)
        self.conv3=nn.Conv1d(
            128,128,kernel_size=3,padding=4,dilation=4)
        self.FC1=nn.Linear(128,1)
        self.FC2=nn.Linear(Co.seq_len,Co.pred_len)
    def forward(self,X): # [B,X_L,X_F] in, [B,y_L,y_F] out.
        X=X.transpose(1,2)
        X=self.conv1(X)
        X=self.conv2(X)
        out=self.conv3(X)
        out=out.transpose(1,2)
        out = self.FC1(out).squeeze(-1) 
        out = self.FC2(out).unsqueeze(-1)
        return out

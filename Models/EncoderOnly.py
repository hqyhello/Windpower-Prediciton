import torch
import torch.nn as nn

class Model(nn.Module):
    ''' nn.EncoderOnly:
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.Layer=nn.TransformerEncoderLayer(d_model=128, nhead=8,batch_first=True)
        self.Encoder=nn.TransformerEncoder(self.Layer,num_layers=3)
        self.Enc_embed=nn.Conv1d(Co.f_in,128,kernel_size=3,padding=1)
        self.Enc_pos=PositionalEmbedding()
        self.FC1=nn.Linear(128,1)
        self.FC2=nn.Linear(Co.seq_len,Co.pred_len)  
        self.Relu=nn.ReLU()

        
    def forward(self,X_enc): # [B,L,F] and [B,L,F] in
        X1=self.Enc_embed(X_enc.transpose(1,2)).transpose(1,2)
        X_enc=X1+self.Enc_pos(X_enc)        
        out=self.Encoder(X_enc) #[B,L,F]-->[B,L,C] 只和X_dec有关
        out = self.FC1(out).squeeze(-1) 
        out = self.FC2(out).unsqueeze(-1)         
        return out
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

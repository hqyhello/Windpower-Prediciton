#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from math import sqrt


# #### Attention Family

# In[2]:


class FullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# In[3]:


class AttentionLayer(nn.Module):
    def __init__(self, attention=FullAttention, d_model=128, n_heads=8, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# #### Encoder 

# In[4]:


class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, d_ff=None, dropout=0.1, activation="relu",gate=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention =AttentionLayer()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model*2, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu 
        self.gate=gate
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # 门控机制
        out, gate = torch.chunk(y,2,dim=-1)
        # 残差正则化
        out = self.norm2(x+out)
        if self.gate:
            out = out * torch.sigmoid(gate)
        return out, attn


# In[5]:


class Encoder(nn.Module):
    def __init__(self, conv_layers=None, norm_layer=None,num_encoder_layers=1,gate=False):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList([EncoderLayer(gate=gate) for _ in range(num_encoder_layers)])
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# #### Dncoder 

# In[6]:


class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, d_ff=None,
                 dropout=0.1, activation="relu",gate=False):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = AttentionLayer()
        self.cross_attention = AttentionLayer()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model*2, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.gate=gate

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # 门控机制
        out, gate = torch.chunk(y,2,dim=-1)
        # 残差正则化
        out = self.norm3(x + out)
        if self.gate:
            out = out * torch.sigmoid(gate)
        return out


# In[7]:


class Decoder(nn.Module):
    def __init__(self, norm_layer=None, projection=None,num_decoder_layers=1,gate=False):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(gate=gate) for _ in range(num_decoder_layers)])
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


# ####  Gated Transformer

# In[11]:


class Gated_Transformer(nn.Module):
    def __init__(self,num_encoder_layers=1,num_decoder_layers=1,gate=False):
        super(Gated_Transformer, self).__init__()

        # Encoder and Decoder
        self.encoder = Encoder(num_encoder_layers=num_encoder_layers,gate=gate)
        self.decoder = Decoder(num_decoder_layers=num_decoder_layers,gate=gate)

    def forward(self, X_enc, X_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Encoder and Decoder
        enc_out=X_enc; dec_out=X_dec
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        out=dec_out
        return out,attns


# In[19]:


#----------------------[ 文件转换 ]------------------------------
# 需要保存后再操作,因为无法实时保存。
#!jupyter nbconvert --to script Gated_Transformer.ipynb


# In[18]:


# model=Gated_Transformer(num_encoder_layers=3,num_decoder_layers=3,gate=True)
# X=torch.randn(20,32,128)
# out=model(X,X)


# In[ ]:





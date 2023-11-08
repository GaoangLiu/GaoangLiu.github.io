#!/usr/bin/env python3

# reference https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
import torch
import math 
from torch.nn import functional as F


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim**-0.5

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # query, key, value: (batch_size, num_heads, seq_len, head_dim)
        # mask: (batch_size, seq_len, seq_len)
        # scaled_attention: (batch_size, num_heads, seq_len, head_dim)
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)
        scaled_attention = torch.matmul(query, key.transpose(-1,
                                                             -2)) * self.scaling
        if mask is not None:
            scaled_attention = scaled_attention.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scaled_attention, dim=-1)
        scaled_attention = torch.matmul(attention_weights, value)
        return scaled_attention, attention_weights

    def split_heads(self, x):
        # x: (batch_size, seq_len, d_model)
        # return: (batch_size, num_heads, seq_len, head_dim)
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2)

    def concat_heads(self, x):
        # x: (batch_size, num_heads, seq_len, head_dim)
        # return: (batch_size, seq_len, d_model)
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                    num_heads * head_dim)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            query, key, value, mask)
        scaled_attention = self.concat_heads(scaled_attention)
        scaled_attention = self.W_o(scaled_attention)
        return scaled_attention, attention_weights


class PositionWisefeedForward(torch.nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWisefeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.W_1 = torch.nn.Linear(d_model, d_ff)
        self.W_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.W_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.W_2(x)
        return x
    
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :]
        pe = pe.unsqueeze(0)
        return x + pe    

class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWisefeedForward(d_model, d_ff, dropout)
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2    
    

class DecoderLayer(torch.nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWisefeedForward(d_model, d_ff, dropout)
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # x: (batch_size, seq_len, d_model)
        # enc_output: (batch_size, seq_len, d_model)
        # look_ahead_mask: (batch_size, seq_len, seq_len)
        # padding_mask: (batch_size, seq_len, seq_len)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output,
                                               padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3, attn_weights_block1, attn_weights_block2
    
    
class Encoder(torch.nn.Module):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 input_vocab_size,
                 max_len=5000,
                 dropout=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.max_len = max_len
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = torch.nn.Dropout(dropout)
        self.enc_layers = torch.nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        # mask: (batch_size, seq_len, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x
    

class Decoder(torch.nn.Module):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 target_vocab_size,
                 max_len=5000,
                 dropout=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.target_vocab_size = target_vocab_size
        self.max_len = max_len
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = torch.nn.Dropout(dropout)
        self.dec_layers = torch.nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,
                x,
                enc_output,
                look_ahead_mask=None,
                padding_mask=None):
        # x: (batch_size, seq_len)
        # enc_output: (batch_size, seq_len, d_model)
        # look_ahead_mask: (batch_size, seq_len, seq_len)
        # padding_mask: (batch_size, seq_len, seq_len)
        seq_len = x.size(1)
        attention_weights = {}
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output,
                                                   look_ahead_mask,
                                                   padding_mask)
            attention_weights['decoder_layer{}_block1'.format(
                i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(
                i + 1)] = block2
        return x, attention_weights
    
    
class Transformer(torch.nn.Module):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 input_vocab_size,
                 target_vocab_size,
                 max_len=5000,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_len = max_len
        self.dropout = dropout

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff,
                               input_vocab_size, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff,
                               target_vocab_size, max_len, dropout)
        self.final_layer = torch.nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_padding_mask=None, look_ahead_mask=None,
                dec_padding_mask=None):
        # inp: (batch_size, seq_len)
        # tar: (batch_size, seq_len)
        # enc_padding_mask: (batch_size, seq_len, seq_len)
        # look_ahead_mask: (batch_size, seq_len, seq_len)
        # dec_padding_mask: (batch_size, seq_len, seq_len)
        enc_output = self.encoder(inp, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights
    
    
class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
        
        def __init__(self, d_model, warmup_steps=4000):
            self.d_model = d_model
            self.warmup_steps = warmup_steps
            super(CustomSchedule, self).__init__(None)
    
        def get_lr(self):
            step_num = max(self.last_epoch, 1)
            arg1 = step_num**-0.5
            arg2 = step_num * (self.warmup_steps**-1.5)
            return [self.d_model**-0.5 * min(arg1, arg2) for _ in self.base_lrs]
        
    
    
# Borrowed from https://github.com/EvelynFan/FaceFormer/blob/main/main.py
import torch
import torch.nn as nn
import math


# Temporal Bias
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1), period, rounding_mode='floor')
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    '''
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0'''
    
    for i in range(T):
        mask[i, i * 2:i * 2 + 2] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
'''
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        print("x", x.shape)
        print("pe", self.pe.shape)
        print("p", self.pe[:, :x.size(1), :].shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)'''

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=4, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize positional encoding for one period (4 time steps)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, period, d_model)
        self.period = period
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Get the input sequence length
        seq_len = x.size(1)

        # Dynamically repeat positional encoding based on input length
        repeat_num = (seq_len // self.period) + 1
        pe = self.pe
        #print("p", pe.shape)
        pe_repeated = pe.repeat(1, repeat_num, 1)  # Repeat along the sequence dimension

        # Slice to match the input sequence length
        pe_repeated = pe_repeated[:, :seq_len, :]

        # Add positional encoding to the input tensor
        #print("x", x.shape)
        #print("pe", pe_repeated.shape)
        x = x + pe_repeated
        return self.dropout(x)

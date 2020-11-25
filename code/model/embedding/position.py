import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=366):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len+1, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)         # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # [d_model/2,]

        pe[1:, 0::2] = torch.sin(position * div_term)   # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)   # broadcasting to [max_len, d_model/2]

        self.register_buffer('pe', pe)

    def forward(self, doy):
        return self.pe[doy, :]


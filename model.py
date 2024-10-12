import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torch
import numpy as np
import math
from typing import Optional, Tuple


class Lightning(nn.Module):
    def __init__(self, input_size, output_size, contextlen, device, **kwargs):
        super(Lightning, self).__init__()
        divisor = 2 if input_size % 2 == 0 else 1

        self.encoder = Encoder(input_size, input_size//divisor, device)
        self.positionalencoder = PositionalEncoding(input_size//divisor, device)
        self.attention = MultiheadDiffAttention(input_size//divisor, input_size//(2*divisor), device=device)

        self.contextlen = contextlen

        self.actions = nn.Sequential(
            nn.Linear(input_size//divisor, input_size),
            nn.ReLU(True),
            nn.Linear(input_size, input_size*divisor),
            nn.ReLU(True),
            nn.Linear(input_size*divisor, output_size)
        ).to(device)
        self.critic = nn.Sequential(
            nn.Linear(input_size//divisor, input_size),
            nn.ReLU(True),
            nn.Linear(input_size, input_size*divisor),
            nn.ReLU(True),
            nn.Linear(input_size*divisor, 1)
        ).to(device)

        self.state_dim = input_size
        self.action_dim = output_size
        self.device = device

    def forward(self, x, critic=False):
        x = x.to(self.device)
        x = self.encoder(x)
        x = self.positionalencoder(x)
        max_len = min(x.size(0), self.contextlen)
        attention_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool().to(self.device)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = x.squeeze(1)
        if critic:
            x = self.critic(x)
            x = x[-1]
        else:
            x = self.actions(x)
            x = x[-1]
            x = F.softmax(x, dim=0)
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, device=torch.device("cpu"), **kwargs):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, input_size*4),
            nn.ReLU(True),
            nn.Linear(input_size*4, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, output_size)
        ).to(device)

    def forward(self, x):
        x = x.to(self.encoder[0].weight.device)
        return self.encoder(x)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device=torch.device("cpu"), max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiheadDiffAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                         kdim, vdim, batch_first, device, dtype)

        # Initialize lambda as a learnable parameter
        self.lambda_param = nn.Parameter(torch.tensor(0.8))

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:

        q, k, v = self._get_input_buffer(query, key, value)

        q1, q2 = torch.chunk(q, 2, dim=-1)
        k1, k2 = torch.chunk(k, 2, dim=-1)

        attn_output_weights1 = torch.matmul(q1, k1.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_output_weights2 = torch.matmul(q2, k2.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_output_weights1 = F.softmax(attn_output_weights1, dim=-1)
        attn_output_weights2 = F.softmax(attn_output_weights2, dim=-1)

        diff_attn_weights = attn_output_weights1 - self.lambda_param * attn_output_weights2

        if attn_mask is not None:
            diff_attn_weights += attn_mask

        if key_padding_mask is not None:
            diff_attn_weights = diff_attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        attn_output = torch.matmul(diff_attn_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(query.shape[0], -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            if average_attn_weights:
                diff_attn_weights = diff_attn_weights.mean(dim=1)
            return attn_output, diff_attn_weights
        else:
            return attn_output, None

    def _get_input_buffer(self, query, key, value):
        if self._qkv_same_embed_dim:
            return F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            q = F.linear(query, self.q_proj_weight, self.in_proj_bias[:self.embed_dim] if self.in_proj_bias is not None else None)
            k = F.linear(key, self.k_proj_weight, self.in_proj_bias[self.embed_dim:(self.embed_dim * 2)] if self.in_proj_bias is not None else None)
            v = F.linear(value, self.v_proj_weight, self.in_proj_bias[(self.embed_dim * 2):] if self.in_proj_bias is not None else None)
            return q, k, v

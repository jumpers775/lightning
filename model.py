import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torch.jit as jit
import torch
import numpy as np
import math
from typing import Optional, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class Lightning(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int,
        last_layer_dim_vf: int,
        contextlen: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.features = feature_dim
        self.contextlen = contextlen

        divisor = 2

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//divisor),
            nn.ReLU(True),
            nn.Linear(feature_dim//divisor, feature_dim//divisor),
            nn.ReLU(True)
        )

        self.positionalencoder = PositionalEncoding(feature_dim//divisor)

        self.attention = MultiheadDiffAttention(feature_dim//divisor, 4)

        self.norm1 = nn.LayerNorm(feature_dim//divisor)
        self.dropout1 = nn.Dropout(0.1)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim//divisor, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim//divisor)
        )

        self.norm2 = nn.LayerNorm(feature_dim//divisor)
        self.dropout2 = nn.Dropout(0.1)

        self.actions = nn.Sequential(
            nn.Linear(feature_dim//divisor, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim*divisor),
            nn.ReLU(True),
            nn.Linear(feature_dim*divisor, self.latent_dim_pi)
        )

        self.critic = nn.Sequential(
            nn.Linear(feature_dim//divisor, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim*divisor),
            nn.ReLU(True),
            nn.Linear(feature_dim*divisor, self.latent_dim_vf)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_generic(self, features: torch.Tensor) -> torch.Tensor:
        x = features.to(self.device)
        batchsize = x.size(0)
        y = x.view(batchsize, self.contextlen, self.features).view(-1, self.features)
        y = self.encoder(y).view(batchsize, self.contextlen, -1)
        y = self.positionalencoder(y)
        z, _ = self.attention(y, y, y)
        if self.training:
            z = self.dropout1(z)
        y = y + z
        y = self.norm1(y)
        z = self.ffn(y)
        if self.training:
            z = self.dropout2(z)
        y = y + z
        y = self.norm2(y)
        return y.reshape(-1, y.size(-1)), batchsize

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        y, batchsize = self.forward_generic(features)
        y = self.actions(y).view(batchsize, self.contextlen, -1)
        return y[:, -1, :]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        y, batchsize = self.forward_generic(features)
        y = self.critic(y).view(batchsize, self.contextlen, -1)
        return y[:, -1, :]


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
        batch_size, seq_len, _ = x.size()
        return x + self.pe[:seq_len, :].expand(batch_size, -1, -1)

class MultiheadDiffAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                         kdim, vdim, batch_first, device, dtype)

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

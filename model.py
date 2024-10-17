import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch import Tensor
import torch.jit as jit
import torch
import numpy as np
import math
from typing import Optional, Tuple

class SimBa(nn.Module):
    def __init__(self, obs_shape, action_shape, device: str ="cpu"):
        super(SimBa, self).__init__()
        self.device = torch.device(device)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.rsnorm = RSNorm(obs_shape)
        self.linear1 = nn.Linear(obs_shape, 256).to(device)
        self.layernorm1 = nn.LayerNorm(256, device=device)
        self.linear2 = nn.Linear(256, 256).to(device)
        self.linear3 = nn.Linear(256, 256).to(device)
        self.layernorm2 = nn.LayerNorm(256, device=device)
        self.outputlayer = nn.Linear(256, action_shape).to(device)
    def forward(self, x):
        x = x.to(self.device)
        x = self.rsnorm(x)
        x = self.linear1(x)
        y = self.layernorm1(x)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear3(y)
        x = x + y
        x = self.layernorm2(x)
        x = self.outputlayer(x)
        x = F.softmax(x, dim=-1)
        return x



class ViViTAE(nn.Module):
    def __init__(self, input_size, output_size, patchnum, maxlen, hidden_dim: int = 512, device: str = "cpu", **kwargs):
        super(ViViTAE, self).__init__()
        self.device = torch.device(device)
        self.tokenizer = nn.Linear(input_size, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.patchembedding = SinusoidalPositionalEmbedding(patchnum, hidden_dim)
        self.spatialencoder = TransformerEncoder(hidden_dim, hidden_dim, device=device)
        self.spatialembedding = SinusoidalPositionalEmbedding(patchnum, hidden_dim)
        self.temporalencoder = TransformerEncoder(hidden_dim, hidden_dim, device=device)
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_size)
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)

    def forward(self, x):
        x = x.to(self.device)
        y = self.tokenizer(x)
        if self.training:
            y = self.dropout(y)
        y = self.patchembedding(y)
        y = self.spatialencoder(y)
        y = self.spatialembedding(y)
        y = self.temporalencoder(y)
        y = self.latent_encoder(y)
        y = self.decoder(y)
        return y




class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, device=torch.device("cpu"), **kwargs):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiheadDiffAttention(input_size, 8, device=device)
        self.norm1 = nn.LayerNorm(input_size, device=device)
        self.dropout1 = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(True),
            nn.Linear(output_size, input_size)
        ).to(device)
        self.norm2 = nn.LayerNorm(input_size, device=device)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x.to(self.encoder[0].weight.device)
        y = self.attention(x, x, x)
        if self.training:
            y = self.dropout1(x)
        x = x + y
        x = self.norm1(x)
        y = self.ffn(x)
        if self.training:
            y = self.dropout2(y)
        x = x + y
        x = self.norm2(x)
        return x

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

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.positional_embedding = self.create_positional_embedding()

    def create_positional_embedding(self):
        position = torch.arange(self.num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.embed_dim))

        pos_emb = torch.zeros(self.num_patches, self.embed_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        return pos_emb.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return x + self.positional_embedding[:, :seq_len, :].expand(batch_size, -1, -1)

class RSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(RSNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            self.num_batches_tracked += 1

            if self.num_batches_tracked == 1:
                update_factor = 1
            else:
                update_factor = self.momentum

            self.running_mean = (1 - update_factor) * self.running_mean + update_factor * batch_mean
            self.running_var = (1 - update_factor) * self.running_var + update_factor * batch_var

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        return (x - mean) / torch.sqrt(var + self.eps)





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

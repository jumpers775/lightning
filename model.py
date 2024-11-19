import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import torch.jit as jit
import torch
import numpy as np
import math
from typing import Optional, Tuple

class SimBa(nn.Module):
    def __init__(self, obs_shape, action_shape, device: str ="cpu", dropout_rate: float = 0.1):
        super(SimBa, self).__init__()
        self.device = torch.device(device)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        # Use RSNorm to normalize inputs for better training stability
        self.rsnorm = RSNorm(obs_shape)
        # Define layers with dimensions scaled by obs_shape to adjust model capacity
        self.linear1 = nn.Linear(obs_shape, obs_shape*8).to(device)
        self.layernorm1 = nn.LayerNorm(obs_shape*8, device=device)
        self.linear2 = nn.Linear(obs_shape*8, obs_shape*16).to(device)
        self.linear3 = nn.Linear(obs_shape*16, obs_shape*8).to(device)
        self.layernorm2 = nn.LayerNorm(obs_shape*8, device=device)
        self.outputlayer = nn.Linear(obs_shape*8, action_shape).to(device)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        x = self.rsnorm(x)  # Normalize input
        x = self.dropout(self.linear1(x))  # Apply first linear layer with dropout

        @torch.compiler.disable(recursive=False)
        def checkpoint_fn(x):
            # Use checkpointing to reduce memory usage during training
            y = self.layernorm1(x)
            y = self.dropout(self.linear2(y))
            y = F.relu(y)
            y = self.dropout(self.linear3(y))
            return y

        y = checkpoint(checkpoint_fn, x, use_reentrant=False)
        x = x + y  # Residual connection to combine input and transformed features
        x = self.layernorm2(x)
        x = self.dropout(self.outputlayer(x))
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, output_size=128, dropout=0.1):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Initial convolutional layers to extract spatial features from input images
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=4)

        # Determine the size after convolutions to configure linear layer
        self.conv_output_size = self._get_conv_output_size(input_size)

        # Reduce dimensions before feeding into LSTM to manage computational load
        self.dim_reduction = nn.Linear(self.conv_output_size, hidden_size)

        # LSTM layers to capture temporal dependencies in sequences
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        seq_len, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)  # Rearrange dimensions for convolutional layers

        x = self.pool(F.relu(self.conv(x)))  # Extract and pool features
        x = self.pool2(x)
        x = x.reshape(seq_len, -1)  # Flatten features for linear layer

        x = self.dropout(F.relu(self.dim_reduction(x)))  # Apply dimension reduction with activation and dropout

        x = x.to(torch.float32)

        @torch.compiler.disable(recursive=False)
        def lstm_forward(x, hidden):
            # Checkpointing to save memory during backpropagation through LSTM
            return self.lstm(x, hidden)

        output, (hidden, cell) = checkpoint(lstm_forward, x, hidden, use_reentrant=False)

        x = self.dropout(self.fc(output))  # Final fully connected layer with dropout

        return x, (hidden, cell)

    def _get_conv_output_size(self, input_size):
        height, width, _ = input_size
        with torch.no_grad():
            # Use a dummy input to calculate the output size after convolutions
            dummy_input = torch.zeros(1, 3, height, width)
            conv_output = self.pool2(self.pool(self.conv(dummy_input)))
            return int(torch.prod(torch.tensor(conv_output.size()[1:])))




class ViViTAE(nn.Module):
    def __init__(self, input_size, output_size, patchnum, maxlen, hidden_dim: int = 512, device: str = "cpu"):
        super(ViViTAE, self).__init__()
        self.device = torch.device(device)
        self.patchnum = patchnum
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim
        self.input_size = input_size

        # Embed image patches to reduce spatial dimensions and capture local features
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patchnum, stride=patchnum)
        # Learnable positional embeddings to retain spatial information
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_size // patchnum) ** 2, hidden_dim))
        # Learnable temporal embeddings to encode sequence order
        self.temporal_embed = nn.Parameter(torch.zeros(1, maxlen, hidden_dim))

        # Transformers to process spatial and temporal features separately
        self.spatial_transformer = TransformerEncoder(hidden_dim, hidden_dim, device=device)
        self.temporal_transformer = TransformerEncoder(hidden_dim, hidden_dim, device=device)

        self.output_proj = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        b, t, c, h, w = x.shape
        t = min(t, self.maxlen)  # Limit the sequence length to maxlen

        spatial_tokens = []
        for i in range(t):
            # Extract patches and add spatial positional embeddings
            patches = self.patch_embed(x[:, i]).flatten(2).transpose(1, 2)
            patches = patches + self.pos_embed[:, :patches.size(1), :]
            # Process spatial features with transformer encoder
            encoded = self.spatial_transformer(patches)
            # Aggregate spatial features by averaging
            spatial_tokens.append(encoded.mean(dim=1, keepdim=True))

        # Concatenate spatial tokens to form a temporal sequence
        temporal_input = torch.cat(spatial_tokens, dim=1)
        temporal_input = temporal_input + self.temporal_embed[:, :t, :]  # Add temporal positional embeddings

        if t < self.maxlen:
            # Pad the temporal sequence if it's shorter than maxlen
            padding = torch.zeros(b, self.maxlen - t, self.hidden_dim, device=self.device)
            temporal_input = torch.cat([temporal_input, padding], dim=1)

        # Process temporal sequence with transformer encoder
        output = self.temporal_transformer(temporal_input)
        return self.output_proj(output)



class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, device=torch.device("cpu"), **kwargs):
        super(TransformerEncoder, self).__init__()
        self.device = device
        # Custom multi-head attention to capture relations with a difference mechanism
        self.attention = MultiheadDiffAttention(input_size, 8, device=device)
        self.norm1 = nn.LayerNorm(input_size, device=device)
        self.dropout1 = nn.Dropout(0.1)
        # Feed-forward network to transform the attention output
        self.ffn = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(True),
            nn.Linear(output_size, input_size)
        ).to(device)
        self.norm2 = nn.LayerNorm(input_size, device=device)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x.to(self.device)
        y = self.attention(x, x, x)
        if self.training:
            y = self.dropout1(x)  # Apply dropout during training for regularization
        x = x + y  # Residual connection to preserve input information
        x = self.norm1(x)
        y = self.ffn(x)
        if self.training:
            y = self.dropout2(y)
        x = x + y  # Residual connection after feed-forward network
        x = self.norm2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device=torch.device("cpu"), max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Precompute positional encodings to add to input embeddings
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # Add positional encoding to input
        return x + self.pe[:seq_len, :].expand(batch_size, -1, -1)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Create fixed positional embeddings using sinusoidal functions
        self.positional_embedding = self.create_positional_embedding()

    def create_positional_embedding(self):
        position = torch.arange(self.num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.embed_dim))

        pos_emb = torch.zeros(self.num_patches, self.embed_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        return pos_emb.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len = x.size()
        # Add positional embedding to input
        return x + self.positional_embedding[:, :seq_len, :].expand(batch_size, -1, -1)

class RSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(RSNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Initialize running statistics for normalization during training
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            self.num_batches_tracked += 1

            # Update running statistics using momentum for moving average
            if self.num_batches_tracked == 1:
                update_factor = 1
            else:
                update_factor = self.momentum

            self.running_mean = (1 - update_factor) * self.running_mean + update_factor * batch_mean
            self.running_var = (1 - update_factor) * self.running_var + update_factor * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics for normalization during evaluation
            mean = self.running_mean
            var = self.running_var
        # Normalize input by removing mean and scaling by variance
        return (x - mean) / torch.sqrt(var + self.eps)




class MultiheadDiffAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                         kdim, vdim, batch_first, device, dtype)

        # Introduce a learnable parameter lambda for adjusting the difference between attention heads
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

        # Get projected queries, keys, and values
        q, k, v = self._get_input_buffer(query, key, value)

        # Split embeddings to compute attention differences
        q1, q2 = torch.chunk(q, 2, dim=-1)
        k1, k2 = torch.chunk(k, 2, dim=-1)

        # Compute attention weights for each split
        attn_output_weights1 = torch.matmul(q1, k1.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_output_weights2 = torch.matmul(q2, k2.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_output_weights1 = F.softmax(attn_output_weights1, dim=-1)
        attn_output_weights2 = F.softmax(attn_output_weights2, dim=-1)

        # Compute difference of attention weights scaled by lambda_param
        diff_attn_weights = attn_output_weights1 - self.lambda_param * attn_output_weights2

        if attn_mask is not None:
            diff_attn_weights += attn_mask

        if key_padding_mask is not None:
            # Apply key padding mask to avoid attending to padding tokens
            diff_attn_weights = diff_attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        # Compute attention output using the difference of attention weights
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
        # Compute linear projections of query, key, and value
        if self._qkv_same_embed_dim:
            return F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            q = F.linear(query, self.q_proj_weight, self.in_proj_bias[:self.embed_dim] if self.in_proj_bias is not None else None)
            k = F.linear(key, self.k_proj_weight, self.in_proj_bias[self.embed_dim:(self.embed_dim * 2)] if self.in_proj_bias is not None else None)
            v = F.linear(value, self.v_proj_weight, self.in_proj_bias[(self.embed_dim * 2):] if self.in_proj_bias is not None else None)
            return q, k, v

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class Lightning(nn.Module):
    def __init__(self, input_size, output_size, contextlen, device, **kwargs):
        super(Lightning, self).__init__()
        divisor = 2 if input_size % 2 == 0 else 1

        self.encoder = Encoder(input_size, input_size//divisor, device)
        self.positionalencoder = PositionalEncoding(input_size//divisor, device)
        self.attention = nn.MultiheadAttention(input_size//divisor, input_size//(2*divisor), device=device)

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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class Lightning(nn.Module):
    def __init__(self, input_size, output_size, device, **kwargs):
        super(Lightning, self).__init__()
        self.encoder = AutoEncoder(input_size, input_size//2, device)
        self.positionalencoder = PositionalEncoding(input_size//2, device)
        self.attention = nn.MultiheadAttention(input_size//2, input_size//4, device=device)
        self.actions = nn.Sequential(
            nn.Linear(input_size//2, input_size),
            nn.ReLU(True),
            nn.Linear(input_size, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, output_size)
        ).to(device)
        self.critic = nn.Sequential(
            nn.Linear(input_size//2, input_size),
            nn.ReLU(True),
            nn.Linear(input_size, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, 1)
        ).to(device)

        self.device = device

    def forward(self, x, critic=False):
        x = x.to(self.device)
        x = self.encoder.encode(x)
        x = self.positionalencoder(x)
        attention_mask = torch.triu(torch.ones(x.size(0), x.size(0)), diagonal=1).bool().to(self.device)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        if critic:
            x = self.critic(x)
        else:
            x = self.actions(x)
        x = x[-1]
        x = F.softmax(x, dim=0)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, device=torch.device("cpu"), **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, input_size*4),
            nn.ReLU(True),
            nn.Linear(input_size*4, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, output_size)
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(output_size, input_size),
            nn.ReLU(True),
            nn.Linear(input_size, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, input_size*4),
            nn.ReLU(True),
            nn.Linear(input_size*4, input_size*2),
            nn.ReLU(True),
            nn.Linear(input_size*2, input_size)
        ).to(device)

    def forward(self, x):
        x.to(self.encoder[0].weight.device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def encode(self, x):
        x.to(self.encoder[0].weight.device)
        return self.encoder(x)
    def decode(self, x):
        x.to(self.decoder[0].weight.device)
        return self.decoder(x)

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

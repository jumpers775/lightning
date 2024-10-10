import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PPO import PPO
from model import Lightning

args = argparse.ArgumentParser()
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--epochs', type=int, default=10)
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args.add_argument('--imitation', type=bool, default=False)
args.add_argument('--model', type=str, default='model.pth')

args = args.parse_args()
try:
    args.device = torch.device(args.device)
except:
    args.device = torch.device('cpu')

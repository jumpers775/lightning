import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import List, Dict, Any


class PPO:
    def __init__(self, network,lr=1e-4, gamma=0.99, K=3, eps_clip=0.2, policysize=1024, valuesize=1024, device=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K = K
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = network
        state_dim = network.state_dim
        action_dim = network.action_dim

        self.autoencoder_optimizer = torch.optim.Adam(self.network.encoder.parameters(), lr=1e-3)

        self.optimizer = Adam(self.network.parameters(), lr=lr)

        self.loss = nn.MSELoss()
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.network(state)
        dist = torch.distributions.Categorical(logits=action_probs)
        action = dist.sample().detach()
        log_probs = dist.log_prob(action).detach()
        action = action.cpu().numpy()
        return np.array([action]).astype(np.int32), log_probs

    def learn(self, states, actions, rewards, next_states, dones, log_probs):
        states = [torch.FloatTensor(np.array(s, dtype=np.float32)).to(self.device) for s in states]
        next_states = [torch.FloatTensor(np.array(s, dtype=np.float32)).to(self.device) for s in next_states]

        actions = torch.tensor(np.array(actions, dtype=np.int32)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)
        log_probs = torch.FloatTensor(np.array(log_probs, dtype=np.float32)).to(self.device)


        lambda_ = 0.95

        with torch.no_grad():
            values = torch.cat([self.network(s, critic=True) for s in states])
            next_values = torch.cat([self.network(s, critic=True) for s in next_states])

            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * lambda_ * (1 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()

        for _ in range(self.K):
            self.optimizer.zero_grad()

            # Policy loss with PPO clipping
            action_probs = torch.cat([self.network(s) for s in states])
            dist = torch.distributions.Categorical(logits=action_probs)
            old_log_probs = log_probs
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_preds = torch.cat([self.network(s, critic=True) for s in states])
            value_loss = self.loss(value_preds, returns)

            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)

            self.optimizer.step()
    def save_model(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

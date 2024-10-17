import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class PPO:
    def __init__(self, actor, critic, lr=3e-4, gamma=0.99, epochs=10, eps_clip=0.2,
                 entropy_coef=0.01, value_coef=0.5, buffer_capacity=10000, batch_size=64, device=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lambda_ * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        for _ in range(self.epochs):
            batch = self.buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, old_log_probs = zip(*batch)

            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

            # Compute returns and advantage
            with torch.no_grad():
                values = self.critic(states).squeeze()
                next_values = self.critic(next_states).squeeze()

            returns = self.compute_gae(rewards, values, next_values, dones)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = returns - values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get new log probs
            new_action_probs = self.actor(states)
            dist = Categorical(new_action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute ratio and surrogates
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

            # Compute actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), returns)

            # Compute total loss
            loss = actor_loss - self.entropy_coef * entropy + self.value_coef * critic_loss

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
        a = actor_loss.item()
        b = critic_loss.item()
        c = entropy.item()
        return a, b, c

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer.push(state, action, reward, next_state, done, log_prob)

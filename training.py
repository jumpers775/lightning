import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from collections import deque
import random

class PPO:
    def __init__(self, actor, critic, action_space, lr=1e-6, gamma=0.99, K=3, eps_clip=0.2, device=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K = K
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policynet = torch.compile(actor.to(self.device))
        self.valuenet = torch.compile(critic.to(self.device))
        self.optimizer = Adam(list(self.policynet.parameters()) + list(self.valuenet.parameters()), lr=lr)
        self.loss = nn.MSELoss()

        self.is_continuous = hasattr(action_space, 'high')
        if self.is_continuous:
            self.action_dim = action_space.shape[0]
            self.action_low = torch.FloatTensor(action_space.low).to(self.device)
            self.action_high = torch.FloatTensor(action_space.high).to(self.device)
        else:
            self.action_dim = action_space.n

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            if self.is_continuous:
                action_mean = self.policynet(state)
                action_std = torch.ones_like(action_mean) * 0.2
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                action = torch.clamp(action, self.action_low, self.action_high)
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                action_probs = F.softmax(self.policynet(state), dim=-1)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob.cpu().numpy()

    def learn(self, states, actions, rewards, next_states, dones, old_log_probs):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)

        # GAE parameters
        lambda_ = 0.95

        with torch.no_grad():
            values = self.valuenet(states).squeeze()
            next_values = self.valuenet(next_states).squeeze()

            # GAE
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * lambda_ * (1 - dones[t]) * gae
                advantages[t] = gae

            # Compute returns from advantages
            returns = advantages + values

        # Normalizing the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.K):
            self.optimizer.zero_grad()

            # Policy loss with PPO clipping
            if self.is_continuous:
                action_mean = self.policynet(states)
                action_std = torch.ones_like(action_mean) * 0.1
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().mean()
            else:
                action_probs = F.softmax(self.policynet(states), dim=-1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions.squeeze().long())
                entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = self.loss(self.valuenet(states).squeeze(), returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            # Compute gradient and apply gradient clipping
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(list(self.policynet.parameters()) + list(self.valuenet.parameters()), max_norm=0.5)

            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / self.K
        return avg_loss

    def train(self):
        self.policynet.train()
        self.valuenet.train()

    def eval(self):
        self.policynet.eval()
        self.valuenet.eval()


class LSTMTrainer:
    def __init__(self, model, learning_rate=0.1, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.compile(model.to(self.device))
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.hidden = (torch.zeros(4, 256).to(self.device),
                       torch.zeros(4, 256).to(self.device))
        self.scaler = GradScaler()

    def learn(self, images, states):
        self.model.train()

        images_tensor = torch.tensor(np.array(images)).to(self.device)
        states_tensor = torch.tensor(np.array(states)).to(self.device)

        shape = images_tensor.shape
        images_tensor = images_tensor.reshape(-1) / 255.0
        images_tensor = images_tensor.reshape(shape).float()

        sequence_length, channels, height, width = images_tensor.shape

        hidden = torch.zeros(4, self.model.hidden_size).to(self.device)
        cell = torch.zeros(4, self.model.hidden_size).to(self.device)

        with autocast(device_type=self.device):
            predicted_states, (hidden, cell) = self.model(images_tensor, (hidden, cell))

            loss = self.criterion(predicted_states, states_tensor.float())

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()

    def evaluate_episode(self, images, states):
        self.model.eval()
        with torch.no_grad():
            images_tensor = torch.stack(images).unsqueeze(0).to(self.device)
            states_tensor = torch.stack(states).unsqueeze(0).to(self.device)

            predicted_states, _ = self.model(images_tensor)
            loss = self.criterion(predicted_states, states_tensor)

        return loss.item()

    def infer(self, image):
        """
        Perform inference on a single image, maintaining state across calls.

        :param image: A single image tensor of shape (210, 160, 3)
        :return: Predicted state tensor
        """
        self.model.eval()
        with torch.no_grad():
            image_tensor = torch.tensor(image).unsqueeze(0).to(self.device)

            image_tensor = image_tensor.float() / 255.0

            predicted_state, self.hidden = self.model(image_tensor, self.hidden)

            return predicted_state.squeeze(0).squeeze(0).cpu()

    def reset_inference_state(self):
        """
        Reset the internal state used for inference.
        Call this method when you want to start a new sequence.
        """
        self.hidden = (torch.zeros(1, self.model.hidden_size).to(self.device),
                       torch.zeros(1, self.model.hidden_size).to(self.device))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

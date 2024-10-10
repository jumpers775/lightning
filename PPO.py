import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPO:
    def __init__(self, network, device, clip_epsilon=0.2, c1=0.5, c2=0.01):
        self.network = network
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss()
        self.autoencoder_optimizer = optim.Adam(
            self.network.encoder.parameters(),
            lr=1e-4
        )
        self.autoencoder_loss_fn = self.loss_fn

    def train(self, states_list, actions, rewards, next_states_list, dones, old_log_probs):
        assert len(states_list) == len(next_states_list) == len(actions) == len(rewards) == len(dones) == len(old_log_probs)

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        all_values = []
        all_next_values = []
        all_new_log_probs = []
        all_entropies = []

        for states, next_states in zip(states_list, next_states_list):
            states_tensor = torch.cat(states).to(self.device)
            next_states_tensor = torch.cat(next_states).to(self.device)

            self.autoencoder_optimizer.zero_grad()
            reconstructed = self.network.encoder(states_tensor)
            autoencoder_loss = self.autoencoder_loss_fn(reconstructed, states_tensor)
            autoencoder_loss.backward()
            self.autoencoder_optimizer.step()

            with torch.no_grad():
                values = self.network(states_tensor, critic=True)
                next_values = self.network(next_states_tensor, critic=True)
                all_values.append(values)
                all_next_values.append(next_values)

            new_log_probs = torch.log(self.network(states_tensor))
            all_new_log_probs.append(new_log_probs)

            probs = self.network(states_tensor)
            entropy = -(probs * torch.log(probs)).sum()
            entropy = entropy.unsqueeze(0)
            all_entropies.append(entropy)

        values = torch.cat(all_values)
        next_values = torch.cat(all_next_values)
        new_log_probs = torch.stack(all_new_log_probs)
        entropies = torch.stack(all_entropies)

        advantages = rewards + (1 - dones) * 0.99 * next_values - values

        for _ in range(10):
            self.optimizer.zero_grad()

            new_log_probs_actions = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            ratio = torch.exp(new_log_probs_actions - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(values, rewards + 0.99 * next_values * (1 - dones))
            entropy_loss = -entropies.mean()

            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss

            loss.backward()
            self.optimizer.step()

        return loss.item(), autoencoder_loss.item()

    def predict(self, states):
        with torch.no_grad():
            states_tensor = torch.cat(states).to(self.device)
            action_probs = self.network(states_tensor)
            action = torch.argmax(action_probs)
            action = action.item()
        return action, action_probs.squeeze()

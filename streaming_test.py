import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import ale_py

# Determine device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# cpu is fastest
device = torch.device('cpu')


print("Using device: " + str(device))

class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.epsilon)

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),  # (210,160,3) -> (51,39,16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # (51,39,16) -> (24,18,32)
            nn.ReLU(),
            nn.Flatten()
        )
        self.flat_size = 24 * 18 * 32

        self.fc1 = nn.Linear(self.flat_size, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self.apply(self._sparse_init)
        self.to(device)

    def _sparse_init(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            sparsity = 0.9
            fan_out = module.weight.size(0)
            n = int(sparsity * fan_out)
            indices = np.random.choice(fan_out, n, replace=False)
            nn.init.uniform_(module.weight, -1/np.sqrt(fan_out), 1/np.sqrt(fan_out))
            module.weight.data[indices, ...] = 0
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        state = state.permute(0, 3, 1, 2)
        x = self.conv_layers(state)
        x = F.leaky_relu(self.fc1(x))
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),  # (210,160,3) -> (51,39,16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (51,39,16) -> (24,18,32)
            nn.ReLU(),
            nn.Flatten()
        )
        self.flat_size = 24 * 18 * 32

        self.fc1 = nn.Linear(self.flat_size, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(self._sparse_init)
        self.to(device)

    def _sparse_init(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            sparsity = 0.9
            fan_out = module.weight.size(0)
            n = int(sparsity * fan_out)
            indices = np.random.choice(fan_out, n, replace=False)
            nn.init.uniform_(module.weight, -1 / np.sqrt(fan_out), 1 / np.sqrt(fan_out))
            module.weight.data[indices, ...] = 0
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        state = state.permute(0, 3, 1, 2)
        x = self.conv_layers(state)
        x = F.leaky_relu(self.fc1(x))
        value = self.value_head(x)
        return value

class StreamAC:
    def __init__(self, state_dim, action_dim, gamma=0.99, lambda_=0.9):
        self.policy = PolicyNetwork(action_dim)
        self.value = ValueNetwork()
        self.gamma = gamma
        self.lambda_ = lambda_

        # Parameters for data scaling using Welford's algorithm
        self.state_mean = torch.zeros(state_dim, device=device)
        self.state_var = torch.ones(state_dim, device=device)
        self.state_count = torch.zeros(1, device=device)
        self.reward_mean = torch.tensor(0.0, device=device)
        self.reward_var = torch.tensor(1.0, device=device)
        self.reward_count = torch.tensor(0.0, device=device)

        # Eligibility traces
        self.reset_eligibility_traces()

    def reset_eligibility_traces(self):
        # Initialize eligibility traces for value network parameters
        self.value_eligibility = []
        for param in self.value.parameters():
            self.value_eligibility.append(torch.zeros_like(param.data))
        # Initialize eligibility traces for policy network parameters
        self.policy_eligibility = []
        for param in self.policy.parameters():
            self.policy_eligibility.append(torch.zeros_like(param.data))

    def update_mean_var(self, x, mean, var, count):
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        var += delta * delta2
        return mean, var, count

    def normalize_state(self, state):
        # Convert state to tensor if it's not already
        state = torch.FloatTensor(state).to(device)

        # Ensure state has the correct shape (210, 160, 3)
        if state.shape != (210, 160, 3):
            raise ValueError(f"Expected state shape (210, 160, 3), got {state.shape}")

        # Update running statistics
        self.state_mean, self.state_var, self.state_count = self.update_mean_var(
            state, self.state_mean, self.state_var, self.state_count)

        # Calculate standardization
        state_std = torch.sqrt(self.state_var / self.state_count + 1e-8)
        normalized_state = (state - self.state_mean) / state_std

        return normalized_state

    def scale_reward(self, reward):
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        self.reward_mean, self.reward_var, self.reward_count = self.update_mean_var(
            reward, self.reward_mean, self.reward_var, self.reward_count)
        reward_std = torch.sqrt(self.reward_var / self.reward_count + 1e-8)
        scaled_reward = reward / reward_std
        return scaled_reward

    def select_action(self, state):
        state = self.normalize_state(state)
        action_probs = self.policy(state)
        action = torch.multinomial(action_probs, 1)
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        reward = self.scale_reward(reward)
        action = torch.tensor(action, dtype=torch.long, device=device)

        # Compute TD error δ
        value = self.value(state)
        next_value = self.value(next_state).detach() if not done else torch.tensor(0.0, device=device)
        delta = reward + self.gamma * next_value - value

        # Update eligibility traces for value network
        value_grads = torch.autograd.grad(value, list(self.value.parameters()), retain_graph=True)
        for i, (eligibility, grad) in enumerate(zip(self.value_eligibility, value_grads)):
            self.value_eligibility[i] = self.gamma * self.lambda_ * eligibility + grad

        # Update eligibility traces for policy network
        action_probs = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Ensure entropy is a scalar if necessary
        entropy = entropy.mean()

        policy_objective = log_prob + 0.01 * entropy * delta.sign()
        policy_grads = torch.autograd.grad(policy_objective, list(self.policy.parameters()), retain_graph=True)

        for i, (eligibility, grad) in enumerate(zip(self.policy_eligibility, policy_grads)):
            self.policy_eligibility[i] = self.gamma * self.lambda_ * eligibility + grad

        # Apply ObGD to value network
        self._apply_obgd(self.value, delta, self.value_eligibility, alpha=1.0, kappa=2.0)

        # Apply ObGD to policy network
        self._apply_obgd(self.policy, delta, self.policy_eligibility, alpha=1.0, kappa=3.0)

    def _apply_obgd(self, network, delta, eligibility_traces, alpha, kappa):
        delta_bar = max(abs(delta.item()), 1.0)
        z_norm = sum([torch.sum(torch.abs(z)) for z in eligibility_traces]).item()
        M = alpha * kappa * delta_bar * z_norm
        alpha_hat = alpha / max(M, 1.0)
        with torch.no_grad():
            for param, eligibility in zip(network.parameters(), eligibility_traces):
                param += alpha_hat * delta.squeeze(0) * eligibility

# Training loop
envname = 'ALE/Alien-v5'
env = gym.make(envname, obs_type="rgb")
state_dim = env.observation_space.shape
action_dim = env.action_space.n

agent = StreamAC(state_dim, action_dim)
total_episodes = 300
max_steps_per_episode = 500
best_reward = float('-inf')
episode_rewards = []  # Store rewards for plotting

pbar = tqdm(range(total_episodes))
for episode in pbar:
    state, _ = env.reset()
    agent.reset_eligibility_traces()
    episode_reward = 0.0

    for step in range(max_steps_per_episode):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)
        episode_reward += reward

        state = next_state
        if done:
            break

    episode_rewards.append(episode_reward)  # Record the episode reward
    if episode_reward > best_reward:
        best_reward = episode_reward

    pbar.set_description(f"Episode Reward: {episode_reward:.2f} | Best: {best_reward:.2f}")

# Plot the learning curve
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.grid(True)
plt.savefig('streaming_ac.png')

# test with human graphics
env = gym.make(envname, obs_type="rgb", render_mode="human")
state, _ = env.reset()
episode_reward = 0

for _ in range(1000):
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    episode_reward += reward
    state = next_state
    env.render()
    time.sleep(0.1)
    if done:
        break

print(f"Test episode reward: {episode_reward}")

env.close()

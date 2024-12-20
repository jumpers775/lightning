import os
import platform

if platform.system() == "Darwin":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
#from training import PPO

# Determine device and compilation mode
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

device = torch.device("cpu")

compile_mode = "aot_eager" if device.type == "mps" else "inductor"

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
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.apply(self._sparse_init)
        self.to(device)

    def _sparse_init(self, module):
        if isinstance(module, nn.Linear):
            sparsity = 0.9
            fan_out = module.weight.size(0)
            n = int(sparsity * fan_out)
            indices = np.random.choice(fan_out, n, replace=False)
            nn.init.uniform_(module.weight, -1/np.sqrt(fan_out), 1/np.sqrt(fan_out))
            module.weight.data[indices, ...] = 0
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        action_logits = self.fc2(x)
        return action_logits

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.apply(self._sparse_init)
        self.to(device)

    def _sparse_init(self, module):
        if isinstance(module, nn.Linear):
            sparsity = 0.9
            fan_out = module.weight.size(0)
            n = int(sparsity * fan_out)
            indices = np.random.choice(fan_out, n, replace=False)
            nn.init.uniform_(module.weight, -1/np.sqrt(fan_out), 1/np.sqrt(fan_out))
            module.weight.data[indices, ...] = 0
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        value = self.fc2(x)
        return value

class StreamAC:
    def __init__(self, policy_net, value_net, gamma=0.99, lambda_=0.9):
        self.policy = torch.compile(policy_net, backend=compile_mode)
        self.value = torch.compile(value_net, backend=compile_mode)
        self.gamma = gamma
        self.lambda_ = lambda_

        # Parameters for data scaling using Welford's algorithm
        self.state_mean = torch.zeros(4, device=device)
        self.state_var = torch.ones(4, device=device)
        self.state_count = torch.zeros(1, device=device)
        self.reward_mean = torch.tensor(0.0, device=device)
        self.reward_var = torch.tensor(1.0, device=device)
        self.reward_count = torch.tensor(0.0, device=device)

        # Eligibility traces
        self.reset_eligibility_traces()

    def reset_eligibility_traces(self):
        self.value_eligibility = []
        for param in self.value.parameters():
            self.value_eligibility.append(torch.zeros_like(param.data))
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
        state = torch.FloatTensor(state).to(device)
        self.state_mean, self.state_var, self.state_count = self.update_mean_var(
            state, self.state_mean, self.state_var, self.state_count)
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
        action_logits = self.policy(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, state, action, reward, next_state, done):
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        reward = self.scale_reward(reward)
        action = torch.tensor(action, dtype=torch.long, device=device)

        value = self.value(state)
        next_value = self.value(next_state).detach() if not done else torch.tensor(0.0, device=device)
        delta = reward + self.gamma * next_value - value

        value_grads = torch.autograd.grad(value, list(self.value.parameters()), retain_graph=True)
        for i, (eligibility, grad) in enumerate(zip(self.value_eligibility, value_grads)):
            self.value_eligibility[i] = self.gamma * self.lambda_ * eligibility + grad

        action_logits = self.policy(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        entropy = entropy.mean()

        policy_objective = log_prob + 0.01 * entropy * delta.sign()
        policy_grads = torch.autograd.grad(policy_objective, list(self.policy.parameters()), retain_graph=True)

        for i, (eligibility, grad) in enumerate(zip(self.policy_eligibility, policy_grads)):
            self.policy_eligibility[i] = self.gamma * self.lambda_ * eligibility + grad

        self._apply_obgd(self.value, delta, self.value_eligibility, alpha=1.0, kappa=2.0)
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
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Instantiate policy and value networks
policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)

# Initialize StreamAC agent
agent = StreamAC(policy_net, value_net)

total_episodes = 10000
max_steps_per_episode = 500
best_reward = float('-inf')
episode_rewards = []  # Store rewards for plotting

pbar = tqdm(range(total_episodes))
for episode in pbar:
    state, _ = env.reset()
    agent.reset_eligibility_traces()
    episode_reward = 0.0

    for step in range(max_steps_per_episode):
        action, _ = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)
        episode_reward += reward

        state = next_state
        if done:
            break

    episode_rewards.append(episode_reward)
    if episode_reward > best_reward:
        best_reward = episode_reward

    pbar.set_description(f"Episode Reward: {episode_reward:.2f} | Best: {best_reward:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.grid(True)
plt.savefig('streaming_ac.png')

# test with human graphics
env = gym.make('CartPole-v1', render_mode="human")

for _ in range(5):
    state, _ = env.reset()
    episode_reward = 0
    while True:
        action, _ = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state
        time.sleep(0.1)
        if done:
            break

print(f"Test episode reward: {episode_reward}")

env.close()

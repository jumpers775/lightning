import gymnasium as gym
import numpy as np
import torch
import sys
from multiprocessing.pool import ThreadPool
import copy
from collections import deque
from PPO import PPO
from model import Lightning
import psutil
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_mem(device):
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(device).total_memory

    elif device.type == 'cpu':
        return psutil.virtual_memory().total

    elif device.type == 'mps':
        return psutil.virtual_memory().total

    else:
        return None



def episode_runner(env, ppo):
    state, _ = env.reset()
    max_steps = env.spec.max_episode_steps

    episode_reward = 0
    states = []
    next_states = []
    state_buffer = []
    action_buffer = []
    reward_buffer = []
    next_state_buffer = []
    done_buffer = []
    log_prob_buffer = []

    for step in range(max_steps):
        action, log_prob = ppo.act(state.astype(np.float32))

        next_state, reward, done, truncated, _ = env.step(action[0])
        episode_reward += reward

        states.append(state.astype(np.float32))
        state_buffer.append(states)
        action_buffer.append(action[0])
        reward_buffer.append(reward)
        next_states.append(next_state.astype(np.float32))
        next_state_buffer.append(next_states)
        done_buffer.append(done)
        log_prob_buffer.append(log_prob.item())

        if len(states) == ppo.network.contextlen:
            states.pop(0)
            next_states.pop(0)

        state = next_state

        if done or truncated:
            break

    return episode_reward, state_buffer, action_buffer, reward_buffer, next_state_buffer, done_buffer, log_prob_buffer

def largestfactorial(n,b):
    for i in range(n, 0, -1):
        if b % i == 0:
            return i

def main():
    env = gym.make('LunarLander-v3')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n


    max_steps = env.spec.max_episode_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memory = get_mem(device)

    max_steps = min(env.spec.max_episode_steps, largestfactorial(memory//state_dim//4, memory))


    network = Lightning(state_dim, action_dim, max_steps, device)
    ppo = PPO(network, device=device)

    num_episodes = 1000 * 6 * 4
    update_frequency = 20

    state_buffer = []
    action_buffer = []
    reward_buffer = []
    next_state_buffer = []
    done_buffer = []
    log_prob_buffer = []

    envs = [gym.make('LunarLander-v3') for _ in range(update_frequency)]

    losses = []
    for episode in tqdm(range(num_episodes//update_frequency)):

        with ThreadPool(update_frequency) as pool:
            results = pool.starmap(episode_runner, [(envs[i], ppo) for i in range(update_frequency)])


        for i, result in enumerate(results):
            losses.append(result[0])
            state_buffer.extend(result[1])
            action_buffer.extend(result[2])
            reward_buffer.extend(result[3])
            next_state_buffer.extend(result[4])
            done_buffer.extend(result[5])
            log_prob_buffer.extend(result[6])

        try:
            ppo.learn(
                state_buffer,
                action_buffer,
                reward_buffer,
                next_state_buffer,
                done_buffer,
                log_prob_buffer
            )
        except Exception as e:
            print(e)
            print("Error in learning, skipping update")
            # work around memory-leak
            ppo.save_model("model.pth")
            del ppo
            ppo = PPO(network, device=device)
            ppo.load_model("model.pth")
        state_buffer.clear()
        action_buffer.clear()
        reward_buffer.clear()
        next_state_buffer.clear()
        done_buffer.clear()
        log_prob_buffer.clear()

    env.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Original Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    x = np.arange(len(losses))
    y = np.array(losses)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y, alpha=0.5)
    plt.plot(x, m*x + b, color='red')
    plt.title('Linear Regression of Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('losses_with_regression.png')

if __name__ == "__main__":
    main()

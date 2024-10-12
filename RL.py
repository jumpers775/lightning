import gymnasium as gym
import numpy as np
import torch
import sys
from multiprocessing.pool import ThreadPool
import copy
from collections import deque
from PPO import PPO
from model import Lightning


if "free-threading" not in sys.version:
    print("WARN: Non free-threading Python detected. This script is optimized for free-threading Python versions.")

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


def main():
    env = gym.make('LunarLander-v3')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_steps = env.spec.max_episode_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = Lightning(state_dim, action_dim, max_steps, device)
    ppo = PPO(network, device=device)

    num_episodes = 1000
    update_frequency = 20

    state_buffer = []
    action_buffer = []
    reward_buffer = []
    next_state_buffer = []
    done_buffer = []
    log_prob_buffer = []

    envs = [gym.make('LunarLander-v3') for _ in range(update_frequency)]

    for episode in range(num_episodes//update_frequency):

        with ThreadPool(update_frequency) as pool:
            results = pool.starmap(episode_runner, [(envs[i], ppo) for i in range(update_frequency)])


        for i, result in enumerate(results):
            print(f"Episode {update_frequency*(episode) + i+ 1}, Reward: {result[0]}")
            state_buffer.extend(result[1])
            action_buffer.extend(result[2])
            reward_buffer.extend(result[3])
            next_state_buffer.extend(result[4])
            done_buffer.extend(result[5])
            log_prob_buffer.extend(result[6])


        ppo.learn(
            state_buffer,
            action_buffer,
            reward_buffer,
            next_state_buffer,
            done_buffer,
            log_prob_buffer
        )

        state_buffer.clear()
        action_buffer.clear()
        reward_buffer.clear()
        next_state_buffer.clear()
        done_buffer.clear()
        log_prob_buffer.clear()

        print(f"Episode {(episode + 1)*update_frequency}, Training completed")


    env.close()

if __name__ == "__main__":
    main()

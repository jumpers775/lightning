import gymnasium as gym
import numpy as np
import torch
from collections import deque
from PPO import PPO
from model import Lightning

def main():
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = Lightning(state_dim, action_dim, device)
    ppo = PPO(network, device=device)

    num_episodes = 80
    max_steps = 500
    update_frequency = 20

    state_buffer = []
    action_buffer = []
    reward_buffer = []
    next_state_buffer = []
    done_buffer = []
    log_prob_buffer = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action, log_prob = ppo.act(state.astype(np.float32))

            next_state, reward, done, truncated, _ = env.step(action[0])
            episode_reward += reward

            state_buffer.append(state.astype(np.float32))
            action_buffer.append(action[0])
            reward_buffer.append(reward)
            next_state_buffer.append(next_state.astype(np.float32))
            done_buffer.append(done)
            log_prob_buffer.append(log_prob.item())

            state = next_state

            if done or truncated:
                break

        if (episode + 1) % update_frequency == 0:
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

            print(f"Episode {episode + 1}, Training completed")

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    main()

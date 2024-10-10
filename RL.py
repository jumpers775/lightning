import gymnasium as gym
import numpy as np
from collections import deque
import torch
from PPO import PPO
from model import Lightning

#torch.autograd.set_detect_anomaly(True)


def main():
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = Lightning(state_dim, action_dim, device)
    ppo = PPO(network, device)

    num_episodes = 1000
    max_steps = 500
    update_frequency = 20

    state_buffer = deque(maxlen=update_frequency * max_steps)
    action_buffer = deque(maxlen=update_frequency * max_steps)
    reward_buffer = deque(maxlen=update_frequency * max_steps)
    next_state_buffer = deque(maxlen=update_frequency * max_steps)
    done_buffer = deque(maxlen=update_frequency * max_steps)
    log_prob_buffer = deque(maxlen=update_frequency * max_steps)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        states = [torch.FloatTensor(state).unsqueeze(0)]
        for step in range(max_steps):
            action, action_probs = ppo.predict(states)

            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            state_buffer.append(states)
            action_buffer.append(action)
            reward_buffer.append(reward)
            states = states + [torch.FloatTensor(next_state).unsqueeze(0)]
            next_state_buffer.append(states)
            done_buffer.append(done)

            with torch.no_grad():
                log_prob = torch.log(action_probs[action]).item()
            log_prob_buffer.append(log_prob)

            if done:
                break

        if (episode + 1) % update_frequency == 0:
            ppo_loss, ae_loss = ppo.train(
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

            print(f"Episode {episode + 1}, PPO Loss: {ppo_loss:.4f}, AE Loss: {ae_loss:.4f}")

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    main()

import gymnasium as gym
import torch
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PPO import PPO
from utils import HistoryWrapper
from model import SimBa, ViViTAE
import FIRSTenv

def RL(train: int = 0, model_path: str = None, out: str = None):

    envname = 'FIRSTenv/FIRSTenv_v0'
    contextlen = 500
    env = gym.make(envname)

    #env = HistoryWrapper(env, history_length=contextlen)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_steps = env.spec.max_episode_steps
    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'


    # useful for multi-threading
    # env = make_vec_env(envname, n_envs=os.cpu_count(), wrapper_class=HistoryWrapper, wrapper_kwargs={'history_length': contextlen})

    #encodingmodel = ViViTAE(state_dim, action_dim, max_steps, contextlen, device=device)

    actingmodel = SimBa(state_dim, action_dim, device=device)

    criticmodel = SimBa(state_dim, 1, device=device)

    ppo = PPO(actingmodel, criticmodel, device=device)
    actor_losses = []
    critic_losses = []
    entropies = []

    pbar = tqdm(range(train), desc="Training", unit="episodes")

    for i in pbar:
        state = env.reset()
        state = state[0]
        done = False

        episode_reward = 0

        while not done:
            action, log_prob = ppo.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            done = terminated or truncated

            ppo.store_transition(state, action, reward, next_state, done, log_prob)
            state = next_state
        actor_loss, critic_loss, entropy = ppo.learn()

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        entropies.append(entropy)

        pbar.set_postfix({
            'actor_loss': f'{actor_loss:.4f}',
            'critic_loss': f'{critic_loss:.4f}',
            'entropy': f'{entropy:.4f}',
            'reward': f'{episode_reward:.2f}'
        })


    # run through a visual test
    env = gym.make(envname, render_mode='human')
    state = env.reset()
    state = state[0]
    done = False
    episode_reward = 0
    while not done:
        action, _ = ppo.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        state = next_state
        env.render()
        time.sleep(0.01)

if __name__ == "__main__":
    RL(train=50, model_path=None, out="model.pth")

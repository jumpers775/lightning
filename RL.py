import gymnasium as gym
import torch
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import ale_py
from training import PPO, LSTMTrainer
from utils import HistoryWrapper, CombinedObservationEnv
from model import SimBa, LSTM
import FIRSTenv

def RL(train: int = 0, model_path: str = None, out: str = None):
    torch.set_float32_matmul_precision('high')
    envname = 'FIRSTenv/FIRSTenv_v0'
    envname = "ALE/Alien-v5"
    env = CombinedObservationEnv(envname)

    #env = HistoryWrapper(env, history_length=contextlen)

    state_dim = env.observation_space["ram"].shape[0]

    if type(env.action_space) == gym.spaces.Discrete:
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]
    max_steps = env.spec.max_episode_steps
    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'


    # useful for multi-threading
    # env = make_vec_env(envname, n_envs=os.cpu_count(), wrapper_class=HistoryWrapper, wrapper_kwargs={'history_length': contextlen})



    actingmodel = SimBa(state_dim, action_dim, device=device)

    criticmodel = SimBa(state_dim, 1, device=device)

    ppo = PPO(actingmodel, criticmodel, env.action_space, device=device)

    encodingmodel = LSTM(env.observation_space["rgb"].shape, 256, output_size=state_dim)

    visiontrainer = LSTMTrainer(encodingmodel, device=device) # gpu uses too much memory


    ppo.train()
    visiontrainer.train()

    actorlosses = []
    reconstructionlosses = []
    episode_rewards = []
    reconstruction_convergance = False
    pbar = tqdm(range(train//2), desc="Training Phase 1", unit=" episodes")

    for i in pbar:
        if i % 10 == 0:
            torch.cuda.empty_cache()
        state, info = env.reset()
        done = False

        visionstates = []
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []

        episode_reward = 0

        while not done:
            action, log_prob = ppo.act(state["ram"])

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            episode_reward += reward

            states.append(state["ram"])
            visionstates.append(state["rgb"])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state["ram"])
            dones.append(done)
            log_probs.append(log_prob)

            state = next_state
        actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
        if not reconstruction_convergance:
            reconstructionloss = visiontrainer.learn(visionstates, states)
            reconstructionlosses.append(reconstructionloss)
            # Calculate moving average of reconstruction loss
            window_size = 10
            if len(reconstructionlosses) >= window_size:
                moving_avg = sum(reconstructionlosses[-window_size:]) / window_size
                if len(reconstructionlosses) % window_size == 0:
                    if abs(moving_avg - reconstructionlosses[-1]) < 5:
                        reconstruction_convergance = True
        else:
            reconstructionloss = "Converged"
        pbar.set_postfix({'actor loss': actorloss, 'reconstruction loss': reconstructionloss})
        actorlosses.append(actorloss)
        episode_rewards.append(episode_reward)


    pbar = tqdm(range(train//2), desc="Training Phase 2", unit=" episodes")

    for i in pbar:
        if i % 10 == 0:
            torch.cuda.empty_cache()
        state, info = env.reset()
        done = False

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []

        episode_reward = 0

        while not done:

            recon_state = visiontrainer.infer(state["rgb"])

            action, log_prob = ppo.act(recon_state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            recon_next_state = visiontrainer.infer(next_state["rgb"])

            done = terminated or truncated

            episode_reward += reward

            states.append(recon_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(recon_next_state)
            dones.append(done)
            log_probs.append(log_prob)

            state = next_state
        actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
        pbar.set_postfix({'actor loss': actorloss})
        actorlosses.append(actorloss)
        episode_rewards.append(episode_reward)




    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

    ax1.plot(actorlosses, label='Actor Losses')
    ax1.set_title('Actor Losses')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(reconstructionlosses, label='Reconstruction Losses')
    ax2.set_title('Reconstruction Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()

    ax3.plot(episode_rewards, label='Episode Rewards')
    ax3.set_title('Episode Rewards')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('losses.png')

    input("Press Enter to continue...")

    env = gym.make(envname, obs_type="rgb", render_mode="human")
    ppo.eval()
    visiontrainer.eval()
    for _ in range(5):
        state, info = env.reset()
        done = False
        while not done:
            ramstate = visiontrainer.infer(state)

            action, _ = ppo.act(ramstate)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            env.render()
            time.sleep(0.01)



if __name__ == "__main__":
    RL(train=int(1000), model_path=None, out="model.pth")

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

    envname = 'FIRSTenv/FIRSTenv_v0'
    envname = "ALE/Pong-v5"
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


    encodingmodel = LSTM(env.observation_space["rgb"].shape, 256, state_dim)

    visiontrainer = LSTMTrainer(encodingmodel, device="cpu")

    ppo = PPO(actingmodel, criticmodel, env.action_space, device=device)

    ppo.train()

    losses = []

    pbar = tqdm(range(train), desc="Training", unit=" episodes")

    for i in pbar:
        state, info = env.reset()
        done = False

        visionstates = []
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []

        while not done:
            action, log_prob = ppo.act(state["ram"])

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            states.append(state["ram"])
            visionstates.append(state["rgb"])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state["ram"])
            dones.append(done)
            log_probs.append(log_prob)

            state = next_state
        loss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
        pbar.set_postfix({'loss': loss})
        visiontrainer.train(visionstates, states)
        losses.append(loss)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, label='losses')
    ax.set_title('Losses')
    ax.set_xlabel('Episode')
    ax.set_ylabel('loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig('losses.png')

    input("Press Enter to continue...")

    env = gym.make(envname, obs_type="rgb", render_mode="human")
    ppo.eval()
    for _ in range(5):
        state, info = env.reset()
        done = False
        while not done:

            ramstate = visiontrainer.infer(state["rgb"])

            action, _ = ppo.act(ramstate)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            env.render()
            time.sleep(0.01)



if __name__ == "__main__":
    RL(train=int(500), model_path=None, out="model.pth")

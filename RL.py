import os
import platform

# Set PyTorch environment variables for compatibility with MPS on macOS
if platform.system() == "Darwin":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gymnasium as gym
import torch
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import ale_py
from training import PPO, LSTMTrainer, RLDistiller
from utils import HistoryWrapper, CombinedObservationEnv
from model import SimBa, LSTM
from mlagents_envs.environment import UnityEnvironment

def RL(train: int = 200, model_path: str = None, out: str = "model.pth"):
    """
    Train a PPO agent with an integrated Vision Trainer using a single progress bar.

    Args:
        train (int, optional): Total number of training episodes. Defaults to 200.
        model_path (str, optional): Path to save or load the model. Defaults to None.
        out (str, optional): Output path for saving models and logs. Defaults to "model.pth".
    """
    phases = 3

    # Increase matmul precision to improve numerical accuracy during training
    torch.set_float32_matmul_precision('high')

    envname = "ALE/Alien-v5"
    env = CombinedObservationEnv(envname)

    state_dim = env.observation_space["ram"].shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        is_continuous = False
    else:
        action_dim = env.action_space.shape[0]
        is_continuous = True

    max_steps = env.spec.max_episode_steps

    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    actingmodel = SimBa(state_dim, action_dim, device=device)
    criticmodel = SimBa(state_dim, 1, device=device)
    ppo = PPO(actingmodel, criticmodel, env.action_space, device=device)

    # Use LSTM to reconstruct RAM state from visual observations
    encodingmodel = LSTM(env.observation_space["rgb"].shape, hidden_size=256, output_size=state_dim)
    visiontrainer = LSTMTrainer(encodingmodel, device=device)

    student_lstm = LSTM(
        input_size=env.observation_space["rgb"].shape,
        hidden_size=128,
        num_layers=2,
        output_size=action_dim,
        dropout=0.05
    )

    ppo.train_mode()
    visiontrainer.train()

    actorlosses = []
    reconstructionlosses = []
    distillation_losses = []
    episode_rewards = []
    reconstruction_convergence = False

    # Divide training into three phases and distillation within 'train' episodes
    phases = 3

    phase_max_episodes = train // phases
    remainder = train % phases

    phase1_max_episodes = phase_max_episodes
    phase2_max_episodes = phase_max_episodes
    distill_episodes = phase_max_episodes + remainder

    total_episodes = train

    pbar = tqdm(total=total_episodes, desc="Training Progress", unit=" episodes")
    current_stage = "Phase 1"

    for episode in range(1, train-distill_episodes + 1):
        if not reconstruction_convergence and episode > phase1_max_episodes:
            current_stage = "Phase 2"

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
            if current_stage == "Phase 1":
                # Use RAM state to train policy while collecting visuals for reconstruction
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

            elif current_stage == "Phase 2":
                # Switch to using reconstructed states from visuals for policy
                recon_state = visiontrainer.infer(state["rgb"])
                action, log_prob = ppo.act(recon_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                recon_next_state = visiontrainer.infer(next_state["rgb"])

                states.append(recon_state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(recon_next_state)
                dones.append(done)
                log_probs.append(log_prob)

                state = next_state

        if current_stage == "Phase 1":
            # Train both PPO and VisionTrainer simultaneously
            actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
            reconstructionloss = visiontrainer.learn(visionstates, states)
            reconstructionlosses.append(reconstructionloss)

            # Check if reconstruction loss has converged to transition to Phase 2
            window_size = 10
            if len(reconstructionlosses) >= window_size:
                moving_avg = sum(reconstructionlosses[-window_size:]) / window_size
                if abs(moving_avg - reconstructionlosses[-1]) < 5:
                    reconstruction_convergence = True
                    current_stage = "Phase 2"
                    print(f"\nReconstruction convergence achieved at episode {episode}. Switching to Phase 2.")

            pbar.set_postfix({'actor loss': actorloss, 'reconstruction loss': reconstructionloss})

        elif current_stage == "Phase 2":
            # Continue training PPO with reconstructed states
            actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
            pbar.set_postfix({'actor loss': actorloss})

        actorlosses.append(actorloss)
        episode_rewards.append(episode_reward)

        pbar.update(1)
        pbar.set_description(f"Training Progress - {current_stage}")

    # Use only visual observations for distillation to train student model
    env = env.envs[env.obs_types.index("rgb")]

    distiller = RLDistiller(
        ppo=ppo,
        lstm_trainer=visiontrainer,
        student=student_lstm,
        env=env,
        device=device
    )

    ppo.eval_mode()
    visiontrainer.eval()

    window_size = 10  # Define window size for convergence checking
    for distill_episode in range(1, distill_episodes + 1):
        distill_loss = distiller.distill(num_passes=1)
        distillation_losses.append(distill_loss)

        pbar.update(1)
        pbar.set_description("Training Progress - Distilling")
        pbar.set_postfix({'distillation loss': distill_loss})

        # Check if distillation loss has converged to possibly exit early
        if len(distillation_losses) >= window_size:
            moving_avg = sum(distillation_losses[-window_size:]) / window_size
            if abs(moving_avg - distillation_losses[-1]) < 0.001:
                print(f"\nDistillation convergence achieved at distillation episode {distill_episode}. Exiting early.")
                break

    pbar.close()

    print("\nEvaluating the Distilled Student Model")

    student_model_path = "student_model.pth"
    distiller.studenttrainer.save(student_model_path)

    studenttrainer = LSTMTrainer(student_lstm, learning_rate=0.001, device=device, hidden=(student_lstm.num_layers, student_lstm.hidden_size))
    studenttrainer.load(student_model_path)
    studenttrainer.eval()

    # Plot training metrics to visualize progress and performance
    fig, axes = plt.subplots(4, 1, figsize=(10, 24))

    axes[0].plot(actorlosses, label='Actor Losses')
    axes[0].set_title('Actor Losses')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(reconstructionlosses, label='Reconstruction Losses')
    axes[1].set_title('Reconstruction Losses')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    if distillation_losses:
        axes[2].plot(range(1, len(distillation_losses) + 1), distillation_losses, label='Distillation Losses')
        axes[2].set_title('Distillation Losses')
        axes[2].set_xlabel('Distillation Episode')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No Distillation Loss Data', horizontalalignment='center', verticalalignment='center')
        axes[2].set_title('Distillation Losses')
        axes[2].set_xlabel('Distillation Episode')
        axes[2].set_ylabel('Loss')

    axes[3].plot(episode_rewards, label='Episode Rewards')
    axes[3].set_title('Episode Rewards')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Reward')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    if model_path:
        visiontrainer.save(out)

    input("Press Enter to continue...")

    eval_env = gym.make(envname, obs_type="rgb", render_mode="human")
    visiontrainer.eval()
    studenttrainer.eval()

    eval_episodes = 5
    for ep in range(1, eval_episodes + 1):
        state, info = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action_logits = studenttrainer.infer(state)
            action = torch.argmax(action_logits).item()

            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

            eval_env.render()
            time.sleep(0.01)

        print(f"Evaluation Episode {ep}: Reward = {episode_reward}")

    eval_env.close()

if __name__ == "__main__":
    RL(train=200, model_path="model_checkpoint.pth", out="model_final.pth")

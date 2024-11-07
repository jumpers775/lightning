import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
from training import PPO, LSTMTrainer, RLDistiller  # Import RLDistiller
from utils import HistoryWrapper, CombinedObservationEnv
from model import SimBa, LSTM
import FIRSTenv

def RL(train: int = 200, model_path: str = None, out: str = "model.pth"):
    """
    Train a PPO agent with an integrated Vision Trainer using a single progress bar.

    Args:
        train (int, optional): Total number of training episodes. Defaults to 200.
        model_path (str, optional): Path to save or load the model. Defaults to None.
        out (str, optional): Output path for saving models and logs. Defaults to "model.pth".
    """
    phases = 3

    # Set higher precision for matrix multiplications
    torch.set_float32_matmul_precision('high')

    # Initialize environment with combined observations
    envname = "ALE/Alien-v5"
    env = CombinedObservationEnv(envname)

    # Determine state and action dimensions
    state_dim = env.observation_space["ram"].shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        is_continuous = False
    else:
        action_dim = env.action_space.shape[0]
        is_continuous = True

    max_steps = env.spec.max_episode_steps

    # Select device
    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize models
    actingmodel = SimBa(state_dim, action_dim, device=device)
    criticmodel = SimBa(state_dim, 1, device=device)
    ppo = PPO(actingmodel, criticmodel, env.action_space, device=device)

    encodingmodel = LSTM(env.observation_space["rgb"].shape, hidden_size=256, output_size=state_dim)
    visiontrainer = LSTMTrainer(encodingmodel, device=device)  # GPU might use too much memory

    # Initialize student model (smaller LSTM)
    student_lstm = LSTM(
        input_size=env.observation_space["rgb"].shape,
        hidden_size=128,  # Reduced hidden size
        num_layers=2,     # Reduced number of layers
        output_size=action_dim,
        dropout=0.05      # Reduced dropout
    )

    # Set models to training mode
    ppo.train_mode()
    visiontrainer.train()

    # Initialize tracking lists
    actorlosses = []
    reconstructionlosses = []
    distillation_losses = []
    episode_rewards = []
    reconstruction_convergence = False

    # Define training phase allocations
    phase1_max_episodes = train // phases  # Allocate half of the training to Phase 1
    phase3_max_episodes = phase1_max_episodes  # Allocate the same number of episodes to Phase 3
    phase2_max_episodes = train - (phase1_max_episodes * 2)  # Remaining episodes for Phase 2
    # Phase 3 will be executed after phase1 and phase2

    # Initialize progress bar
    pbar = tqdm(total=train, desc="Training Progress", unit=" episodes")

    current_stage = "Phase 1"  # Initialize current stage

    for episode in range(1, train + 1):
        # Update stage based on episode count
        if not reconstruction_convergence and episode > phase1_max_episodes:
            current_stage = "Phase 2"

        # Reset environment
        state, info = env.reset()
        done = False

        # Initialize episode-specific lists
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
                # Phase 1: Train PPO and Vision Trainer simultaneously
                action, log_prob = ppo.act(state["ram"])

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward

                # Collect experiences
                states.append(state["ram"])
                visionstates.append(state["rgb"])
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state["ram"])
                dones.append(done)
                log_probs.append(log_prob)

                state = next_state

            elif current_stage == "Phase 2":
                # Phase 2: Use Vision Trainer for inference while training PPO
                recon_state = visiontrainer.infer(state["rgb"])
                action, log_prob = ppo.act(recon_state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward

                recon_next_state = visiontrainer.infer(next_state["rgb"])

                # Collect experiences
                states.append(recon_state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(recon_next_state)
                dones.append(done)
                log_probs.append(log_prob)

                state = next_state

        # Learn from collected experiences
        if current_stage == "Phase 1":
            actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
            # Train Vision Trainer
            reconstructionloss = visiontrainer.learn(visionstates, states)
            reconstructionlosses.append(reconstructionloss)

            # Calculate moving average of reconstruction loss
            window_size = 10
            if len(reconstructionlosses) >= window_size:
                moving_avg = sum(reconstructionlosses[-window_size:]) / window_size
                # Check for convergence (threshold can be adjusted)
                if abs(moving_avg - reconstructionlosses[-1]) < 5:
                    reconstruction_convergence = True
                    current_stage = "Phase 2"
                    print(f"\nReconstruction convergence achieved at episode {episode}. Switching to Phase 2.")

            pbar.set_postfix({'actor loss': actorloss, 'reconstruction loss': reconstructionloss})

        elif current_stage == "Phase 2":
            actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
            pbar.set_postfix({'actor loss': actorloss, 'stage': current_stage})

        # Append losses and rewards
        actorlosses.append(actorloss)
        episode_rewards.append(episode_reward)

        # Update progress bar
        pbar.update(1)
        pbar.set_description(f"Training Progress - {current_stage}")

    # Close training phases
    pbar.close()

    # -------------------- Phase 3: Distillation -------------------- #
    print("\nStarting Phase 3: Distillation")

    env = env.envs[env.obs_types.index("rgb")]

    # Initialize Distiller
    distiller = RLDistiller(
        ppo=ppo,
        lstm_trainer=visiontrainer,
        student=student_lstm,
        env=env,
        device=device
    )

    # Load trained teacher and vision trainer models
    ppo.eval_mode()
    visiontrainer.eval()

    # Perform distillation
    distill_episodes = phase3_max_episodes  # Number of episodes for distillation
    distillation_pbar = tqdm(total=distill_episodes, desc="Distillation Progress", unit=" episodes")
    for _ in range(distill_episodes):
        # Run distillation for each episode
        distill_loss = distiller.distill(num_passes=1)
        distillation_losses.append(distill_loss)
        distillation_pbar.update(1)
        # Optionally, track and log distillation loss here

    distillation_pbar.close()

    # Evaluate the distilled student model
    print("\nEvaluating the Distilled Student Model")

    # Save the distilled student model
    student_model_path = "student_model.pth"
    distiller.studenttrainer.save(student_model_path)

    # Load the student model into LSTMTrainer
    studenttrainer = LSTMTrainer(student_lstm, learning_rate=0.001, device=device, hidden=(student_lstm.num_layers, student_lstm.hidden_size))
    studenttrainer.load(student_model_path)
    studenttrainer.eval()

    # Plotting results
    fig, axes = plt.subplots(4, 1, figsize=(10, 24))  # Added a fourth plot for distillation

    # Plot Actor Losses
    axes[0].plot(actorlosses, label='Actor Losses')
    axes[0].set_title('Actor Losses')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot Reconstruction Losses
    axes[1].plot(reconstructionlosses, label='Reconstruction Losses')
    axes[1].set_title('Reconstruction Losses')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    # Plot Distillation Losses (if tracked)
    if distillation_losses:
        axes[2].plot(distillation_losses, label='Distillation Losses')
        axes[2].set_title('Distillation Losses')
        axes[2].set_xlabel('Distillation Episode')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No Distillation Loss Data', horizontalalignment='center', verticalalignment='center')
        axes[2].set_title('Distillation Losses')
        axes[2].set_xlabel('Distillation Episode')
        axes[2].set_ylabel('Loss')

    # Plot Episode Rewards
    axes[3].plot(episode_rewards, label='Episode Rewards')
    axes[3].set_title('Episode Rewards')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Reward')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # Save models if model_path is provided
    if model_path:
        ppo.save_checkpoint(model_path)
        visiontrainer.save(out)  # Assuming LSTMTrainer has a save method

    input("Press Enter to continue...")

    # -------------------- Evaluation Phase -------------------- #
    eval_env = gym.make(envname, obs_type="rgb", render_mode="human")
    visiontrainer.eval()
    studenttrainer.eval()

    eval_episodes = 5
    for ep in range(1, eval_episodes + 1):
        state, info = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Use Student LSTMTrainer in inference mode
            action_logits = studenttrainer.infer(state)
            action = torch.argmax(action_logits).item()

            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

            # Render environment
            eval_env.render()
            time.sleep(0.01)

        print(f"Evaluation Episode {ep}: Reward = {episode_reward}")

    eval_env.close()

    # Optionally, evaluate the student model separately if needed
    # This can be added similarly to the above evaluation loop

if __name__ == "__main__":
    RL(train=300, model_path="model_checkpoint.pth", out="model_final.pth")

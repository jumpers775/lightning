import os
# Enable MPS fallback for PyTorch to ensure compatibility on devices with MPS support
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
from training import PPO, LSTMTrainer, RLDistiller
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

    # Set high precision for float32 matrix multiplication to improve numerical accuracy during training
    torch.set_float32_matmul_precision('high')

    envname = "ALE/Alien-v5"
    # Use a custom environment with combined observations for the specified game
    env = CombinedObservationEnv(envname)

    # Get the dimension of the RAM state for the agent's input size
    state_dim = env.observation_space["ram"].shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        is_continuous = False
    else:
        action_dim = env.action_space.shape[0]
        is_continuous = True

    # Get the maximum steps per episode for the environment
    max_steps = env.spec.max_episode_steps

    # Select computation device (GPU if available, else CPU) and inform the user
    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the actor and critic models for PPO with the appropriate dimensions
    actingmodel = SimBa(state_dim, action_dim, device=device)
    criticmodel = SimBa(state_dim, 1, device=device)
    ppo = PPO(actingmodel, criticmodel, env.action_space, device=device)

    # Prepare the LSTM model to encode visual observations into RAM state dimensions
    encodingmodel = LSTM(env.observation_space["rgb"].shape, hidden_size=256, output_size=state_dim)
    visiontrainer = LSTMTrainer(encodingmodel, device=device)

    # Define a student LSTM model to be trained via distillation later
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

    # Divide the training into three phases for modular training strategy
    phase1_max_episodes = train // phases
    phase3_max_episodes = phase1_max_episodes
    phase2_max_episodes = train - (phase1_max_episodes * 2)

    # Initialize progress bar and set initial training stage to Phase 1
    pbar = tqdm(total=train, desc="Training Progress", unit=" episodes")
    current_stage = "Phase 1"

    for episode in range(1, train + 1):
        # Transition to Phase 2 if reconstruction has not converged after Phase 1 episodes
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
                # In Phase 1, use RAM state for policy and collect both RAM and visual observations
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
                # In Phase 2, use reconstructed states from visual observations for policy
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
            # During Phase 1, train PPO and VisionTrainer simultaneously
            actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
            reconstructionloss = visiontrainer.learn(visionstates, states)
            reconstructionlosses.append(reconstructionloss)

            # Check for convergence of reconstruction loss to transition to Phase 2
            window_size = 10
            if len(reconstructionlosses) >= window_size:
                moving_avg = sum(reconstructionlosses[-window_size:]) / window_size
                if abs(moving_avg - reconstructionlosses[-1]) < 5:
                    reconstruction_convergence = True
                    current_stage = "Phase 2"
                    print(f"\nReconstruction convergence achieved at episode {episode}. Switching to Phase 2.")

            pbar.set_postfix({'actor loss': actorloss, 'reconstruction loss': reconstructionloss})

        elif current_stage == "Phase 2":
            # In Phase 2, continue training PPO with reconstructed states
            actorloss = ppo.learn(states, actions, rewards, next_states, dones, log_probs)
            pbar.set_postfix({'actor loss': actorloss, 'stage': current_stage})

        actorlosses.append(actorloss)
        episode_rewards.append(episode_reward)

        pbar.update(1)
        pbar.set_description(f"Training Progress - {current_stage}")

    pbar.close()

    print("\nStarting Phase 3: Distillation")

    # Switch environment to use only visual observations for distillation
    env = env.envs[env.obs_types.index("rgb")]

    # Initialize distiller with trained PPO and VisionTrainer to train student model
    distiller = RLDistiller(
        ppo=ppo,
        lstm_trainer=visiontrainer,
        student=student_lstm,
        env=env,
        device=device
    )

    # Set models to evaluation mode during distillation
    ppo.eval_mode()
    visiontrainer.eval()

    distill_episodes = phase3_max_episodes
    # Perform distillation over Phase 3 episodes to train the student model
    distillation_pbar = tqdm(total=distill_episodes, desc="Distillation Progress", unit=" episodes")
    for _ in range(distill_episodes):
        distill_loss = distiller.distill(num_passes=1)
        distillation_losses.append(distill_loss)
        distillation_pbar.update(1)

    distillation_pbar.close()

    print("\nEvaluating the Distilled Student Model")

    # Save and load the trained student model for evaluation
    student_model_path = "student_model.pth"
    distiller.studenttrainer.save(student_model_path)

    studenttrainer = LSTMTrainer(student_lstm, learning_rate=0.001, device=device, hidden=(student_lstm.num_layers, student_lstm.hidden_size))
    studenttrainer.load(student_model_path)
    studenttrainer.eval()

    # Plot training metrics to visualize losses and rewards over episodes
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

    axes[3].plot(episode_rewards, label='Episode Rewards')
    axes[3].set_title('Episode Rewards')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Reward')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # Save the trained vision model if a model_path is provided
    if model_path:
        visiontrainer.save(out)

    input("Press Enter to continue...")

    # Set up evaluation environment and switch models to evaluation mode
    eval_env = gym.make(envname, obs_type="rgb", render_mode="human")
    visiontrainer.eval()
    studenttrainer.eval()

    # Evaluate the performance of the student model in the environment
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
    RL(train=300, model_path="model_checkpoint.pth", out="model_final.pth")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from collections import deque
import random

class PPO:
    """
    Proximal Policy Optimization (PPO) Agent with Enhanced Features.

    This class implements the PPO algorithm with support for both discrete and continuous action spaces.
    It incorporates best practices such as Generalized Advantage Estimation (GAE), entropy regularization,
    learning rate scheduling, advantage normalization, and gradient clipping.

    ### Attributes:
        policynet (nn.Module): The policy network (actor).
        valuenet (nn.Module): The value network (critic).
        optimizer (torch.optim.Optimizer): Optimizer for updating policy and value networks.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Learning rate scheduler.
        is_continuous (bool): Flag indicating if the action space is continuous.
        action_dim (int): Dimension of the action space.
        action_low (torch.Tensor or None): Lower bounds for continuous actions.
        action_high (torch.Tensor or None): Upper bounds for continuous actions.
        gamma (float): Discount factor.
        eps_clip (float): Clipping parameter for PPO.
        K (int): Number of optimization epochs per update.
        gae_lambda (float): Lambda parameter for GAE.
        entropy_coef (float): Coefficient for entropy regularization.
        vf_coef (float): Coefficient for value function loss.
        max_grad_norm (float): Maximum norm for gradient clipping.
        advantage_normalization (bool): Whether to normalize advantages.
        device (torch.device): Device to run computations on.
    """

    def __init__(
        self,
        actor,
        critic,
        action_space,
        lr=3e-4,
        gamma=0.99,
        K=4,
        eps_clip=0.2,
        device=None,
        gae_lambda=0.95,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_scheduler=True,
        scheduler_step_size=1000,
        scheduler_gamma=0.99,
        advantage_normalization=True,
    ):
        """
        Initialize the PPO agent.

        Args:
            actor (nn.Module): The policy network.
            critic (nn.Module): The value network.
            action_space (gym.Space): The action space of the environment.
            lr (float, optional): Learning rate. Defaults to 3e-4.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            K (int, optional): Number of optimization epochs per update. Defaults to 4.
            eps_clip (float, optional): PPO clipping parameter. Defaults to 0.2.
            device (torch.device or str, optional): Device to run computations on. Defaults to CUDA if available.
            gae_lambda (float, optional): Lambda for GAE. Defaults to 0.95.
            entropy_coef (float, optional): Entropy regularization coefficient. Defaults to 0.01.
            vf_coef (float, optional): Value function loss coefficient. Defaults to 0.5.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 0.5.
            use_scheduler (bool, optional): Whether to use a learning rate scheduler. Defaults to True.
            scheduler_step_size (int, optional): Step size for the scheduler. Defaults to 1000.
            scheduler_gamma (float, optional): Gamma for the scheduler decay. Defaults to 0.99.
            advantage_normalization (bool, optional): Whether to normalize advantages. Defaults to True.
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K = K
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.advantage_normalization = advantage_normalization

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backend = 'aot_eager' if device == "mps" else "inductor"
        # Compile networks to improve execution speed during training
        self.policynet = torch.compile(actor.to(self.device), backend=backend)
        self.valuenet = torch.compile(critic.to(self.device), backend=backend)

        # Use a single optimizer for both networks to update them together
        self.optimizer = Adam(
            list(self.policynet.parameters()) + list(self.valuenet.parameters()),
            lr=lr
        )

        # Set up learning rate scheduler to adjust the learning rate over time
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
            )
        else:
            self.scheduler = None

        # Use Mean Squared Error loss for value function approximation
        self.value_loss_fn = nn.MSELoss()

        # Check if action space is continuous or discrete to handle accordingly
        self.is_continuous = hasattr(action_space, 'high')
        if self.is_continuous:
            self.action_dim = action_space.shape[0]
            self.action_low = torch.FloatTensor(action_space.low).to(self.device)
            self.action_high = torch.FloatTensor(action_space.high).to(self.device)
            # Initialize log standard deviation for continuous action exploration
            self.log_std = nn.Parameter(torch.zeros(self.action_dim)).to(self.device)
        else:
            self.action_dim = action_space.n

    def act(self, state):
        """
        Select an action based on the current policy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            action (np.ndarray): Selected action.
            log_prob (float): Log probability of the selected action.
        """
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            if self.is_continuous:
                action_mean = self.policynet(state)
                action_std = self.log_std.exp().expand_as(action_mean)
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                action = torch.clamp(action, self.action_low, self.action_high)
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                action_logits = self.policynet(state)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob.cpu().numpy()

    def process(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.policynet(state)

    def learn(self, states, actions, rewards, next_states, dones, old_log_probs):
        """
        Update the policy and value networks using collected experiences.

        Args:
            states (list or np.ndarray): List of states.
            actions (list or np.ndarray): List of actions taken.
            rewards (list or np.ndarray): List of rewards received.
            next_states (list or np.ndarray): List of next states.
            dones (list or np.ndarray): List of done flags.
            old_log_probs (list or np.ndarray): List of log probabilities of actions taken.

        Returns:
            avg_loss (float): Average loss over optimization epochs.
        """
        # Convert inputs to tensors for PyTorch computations
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)

        # Compute current and next value estimates needed for advantage computation
        with torch.no_grad():
            values = self.valuenet(states).squeeze()
            next_values = self.valuenet(next_states).squeeze()

            # Calculate temporal difference errors (deltas) for GAE
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            # Compute advantages using GAE to reduce variance in policy gradient estimation
            advantages = torch.zeros_like(rewards).to(self.device)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae

            # Calculate returns by adding advantages to value estimates for training critic
            returns = advantages + values

        # Normalize advantages for stability during training
        if self.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0

        for _ in range(self.K):
            # Compute new action probabilities for PPO update
            if self.is_continuous:
                action_mean = self.policynet(states)
                action_std = self.log_std.exp().expand_as(action_mean)
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
            else:
                action_logits = self.policynet(states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions.squeeze().long())
                entropy = dist.entropy().mean()

            # Calculate ratio of new and old action probabilities for PPO clipping
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Compute clipped surrogate objective for policy
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value function loss to update critic
            value_predictions = self.valuenet(states).squeeze()
            value_loss = self.value_loss_fn(value_predictions, returns)

            # Combine all components into total loss
            loss = (policy_loss
                    + self.vf_coef * value_loss
                    - self.entropy_coef * entropy)

            # Perform backpropagation to update policy and value networks
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(list(self.policynet.parameters()) +
                                     list(self.valuenet.parameters()) +
                                     ([self.log_std] if self.is_continuous else []),
                                     self.max_grad_norm)

            # Optimizer step to update parameters
            self.optimizer.step()

            # Update learning rate scheduler if used
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / self.K
        return avg_loss

    def train_mode(self):
        """
        Set the policy and value networks to training mode.
        """
        self.policynet.train()
        self.valuenet.train()

    def eval_mode(self):
        """
        Set the policy and value networks to evaluation mode.
        """
        self.policynet.eval()
        self.valuenet.eval()

    def save_checkpoint(self, filepath):
        """
        Save the model checkpoints.

        Args:
            filepath (str): Path to save the checkpoint.
        """
        checkpoint = {
            'policynet_state_dict': self.policynet.state_dict(),
            'valuenet_state_dict': self.valuenet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'log_std': self.log_std.state_dict() if self.is_continuous else None
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load the model checkpoints.

        Args:
            filepath (str): Path to load the checkpoint from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policynet.load_state_dict(checkpoint['policynet_state_dict'])
        self.valuenet.load_state_dict(checkpoint['valuenet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.is_continuous and 'log_std' in checkpoint:
            self.log_std.load_state_dict(checkpoint['log_std'])

class LSTMTrainer:
    def __init__(self, model, learning_rate=0.1, device=None, hidden=(4, 256)):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.compile(model.to(self.device), backend='aot_eager' if device == "mps" else "inductor")
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.hidden = (torch.zeros(hidden[0], hidden[1]).to(self.device),
                       torch.zeros(hidden[0], hidden[1]).to(self.device))
        self.scaler = GradScaler()

    def learn(self, images, states):
        self.model.train()

        images_tensor = torch.tensor(np.array(images)).to(self.device)
        states_tensor = torch.tensor(np.array(states)).to(self.device)

        shape = images_tensor.shape
        # Normalize image pixel values to [0, 1] for model input
        images_tensor = images_tensor.reshape(-1) / 255.0
        images_tensor = images_tensor.reshape(shape).float()

        sequence_length, channels, height, width = images_tensor.shape

        # Initialize hidden states for LSTM at each training step
        hidden = torch.zeros(4, self.model.hidden_size).to(self.device)
        cell = torch.zeros(4, self.model.hidden_size).to(self.device)

        with autocast(device_type=self.device):
            predicted_states, (hidden, cell) = self.model(images_tensor, (hidden, cell))

            loss = self.criterion(predicted_states, states_tensor.float())

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()

    def evaluate_episode(self, images, states):
        self.model.eval()
        with torch.no_grad():
            images_tensor = torch.stack(images).unsqueeze(0).to(self.device)
            states_tensor = torch.stack(states).unsqueeze(0).to(self.device)

            predicted_states, _ = self.model(images_tensor)
            loss = self.criterion(predicted_states, states_tensor)

        return loss.item()

    def infer(self, image):
        """
        Perform inference on a single image, maintaining state across calls.

        :param image: A single image tensor of shape (210, 160, 3)
        :return: Predicted state tensor
        """
        self.model.eval()
        with torch.no_grad():
            image_tensor = torch.tensor(image).unsqueeze(0).to(self.device)

            image_tensor = image_tensor.float() / 255.0

            predicted_state, self.hidden = self.model(image_tensor, self.hidden)

            return predicted_state.squeeze(0).squeeze(0).cpu()
    def infer_grad(self, image):
        """
        Perform inference on a single image, maintaining state across calls.

        :param image: A single image tensor of shape (210, 160, 3)
        :return: Predicted state tensor
        """
        self.model.eval()
        image_tensor = torch.tensor(image).unsqueeze(0).to(self.device)

        image_tensor = image_tensor.float() / 255.0

        predicted_state, self.hidden = self.model(image_tensor, self.hidden)

        return predicted_state.squeeze(0).squeeze(0).cpu()
    def reset_inference_state(self):
        """
        Reset the internal state used for inference.
        Call this method when you want to start a new sequence.
        """
        self.hidden = (torch.zeros(1, self.model.hidden_size).to(self.device),
                       torch.zeros(1, self.model.hidden_size).to(self.device))
    def detach_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class RLDistiller:
    def __init__(self, ppo, lstm_trainer, student, env, device=None):
        self.ppo = ppo
        self.lstm_trainer = lstm_trainer
        self.student = student.to(device)
        self.studenttrainer = LSTMTrainer(student, learning_rate=0.001, device=device, hidden=(student.num_layers, student.hidden_size))
        self.env = env
        self.device = device
        self.optimizer = optim.Adam(student.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.finetune_optimizer = optim.Adam(student.parameters(), lr=0.0001)

    def distill(self, num_passes):
        # First phase: Distillation
        # Train student model to mimic PPO outputs using LSTM-encoded states
        self.ppo.eval_mode()
        self.lstm_trainer.eval()
        self.studenttrainer.train()

        total_loss = 0.0
        total_steps = 0

        for _ in range(num_passes):
            state, _ = self.env.reset()
            done = False

            while not done:
                # Use LSTM trainer to encode the state for comparison with student
                encoded_state = self.lstm_trainer.infer(state)

                # Obtain target outputs from the PPO agent for distillation
                with torch.no_grad():
                    output = self.ppo.process(encoded_state)
                    action, _ = self.ppo.act(encoded_state)

                # Compute student model's prediction and loss against PPO output
                student_output = self.studenttrainer.infer_grad(state)
                loss = self.criterion(student_output.to(self.device), output.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_steps += 1

                self.studenttrainer.detach_hidden()

                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state

        # Second phase: Finetuning
        # Finetune the student model with policy gradient updates for improved performance
        finetune_passes = num_passes // 4
        self.studenttrainer.train()

        for _ in range(finetune_passes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                student_output = self.studenttrainer.infer(state["rgb"].unsqueeze(0).float() / 255.0)
                action = student_output.argmax().item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # Update student model using rewards to refine policy
                if reward > 0:
                    advantage = torch.tensor(reward).float().to(self.device)
                    loss = -torch.log(F.softmax(student_output, dim=-1)[action]) * advantage

                    self.finetune_optimizer.zero_grad()
                    loss.backward()
                    self.finetune_optimizer.step()

                    total_loss += loss.item()
                    total_steps += 1

                state = next_state

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        return avg_loss

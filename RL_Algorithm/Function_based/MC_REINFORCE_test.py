from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import matplotlib
import matplotlib.pyplot as plt

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
    ) -> None:
        """
            Initialize the MC-REINFORCE Agent.
        """
        # ========= put your code here ========= #
        self.LR = learning_rate
        self.device = device or torch.device("cpu")
        # Initialize policy network and optimizer
        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.episode_durations = []
        # For live plotting in IPython
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def calculate_stepwise_returns(self, rewards: list[float]) -> torch.Tensor:
        """
        Compute normalized stepwise returns for the trajectory.
        """
        # ========= put your code here ========= #
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns.numel() > 1 :
            std = returns.std(unbiased=False)
            returns = (returns - returns.mean()) / (std + 1e-8)
        else:
            returns = returns - returns.mean()

        return returns
        # ====================================== #

    def generate_trajectory(self, env, max_steps: int = 500):
        """
        Generate a trajectory by interacting with the environment.
        Returns: episode_return, stepwise_returns, log_prob_actions, trajectory
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        # Convert initial observation to tensor
        if isinstance(obs, dict):
            state = obs['policy'].squeeze().to(self.device)
        else:
            state = torch.tensor(obs, dtype=torch.float32, device=self.device)

        log_prob_actions = []
        rewards = []
        trajectory = []
        total_reward = 0.0
        done = False
        t = 0
        # ====================================== #

        while not done and t < max_steps:
            # Predict action from the policy network
            # ========= put your code here ========= #
            state_in = state.unsqueeze(0)  # add batch dimension
            probs = self.policy_net(state_in)
            m = distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action).squeeze(0)
            a = action.item()
            # ====================================== #

            # Execute action in the environment
            # ========= put your code here ========= #
            env_action = self.scale_action(a).unsqueeze(0).to(self.device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            r = float(reward)
            # ====================================== #

            # Store log probability, reward, trajectory
            # ========= put your code here ========= #
            log_prob_actions.append(log_prob)
            rewards.append(r)
            trajectory.append((state.cpu().numpy(), a))
            total_reward += r
            # ====================================== #

            # Update state for next step
            if isinstance(next_obs, dict):
                state = next_obs['policy'].squeeze().to(self.device)
            else:
                state = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            done = terminated or truncated
            t += 1

        # Stack log_prob_actions & compute returns
        # ========= put your code here ========= #
        stepwise_returns = self.calculate_stepwise_returns(rewards)
        log_prob_tensor = torch.stack(log_prob_actions)
        # ====================================== #

        return total_reward, stepwise_returns, log_prob_tensor, trajectory, t

    def calculate_loss(self, stepwise_returns: torch.Tensor, log_prob_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for policy optimization.
        """
        # ========= put your code here ========= #
        loss = -torch.sum(log_prob_actions * stepwise_returns)
        return loss
        # ====================================== #

    def update_policy(self, stepwise_returns: torch.Tensor, log_prob_actions: torch.Tensor) -> float:
        """
        Update the policy using the calculated loss.
        """
        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # ====================================== #

    def learn(self, env):
        """
        Train the agent on a single episode.
        Returns: episode_return, loss, trajectory
        """
        # ========= put your code here ========= #
        self.policy_net.train()
        episode_return, stepwise_returns, log_prob_actions, trajectory, count = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return episode_return, loss, trajectory, count
        # ====================================== #

    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float32)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
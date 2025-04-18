from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from RL_Algorithm.RL_base_function import BaseAlgorithm

def to_tensor(obs, device):
    """Convert environment observation to a torch tensor."""
    if isinstance(obs, dict):
        return obs['policy'].squeeze().to(device)
    return torch.tensor(obs, dtype=torch.float32, device=device)

class Actor(nn.Module):
    """
    Discrete-action policy network for A2C.
    Outputs logits for a categorical distribution over actions.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Critic(nn.Module):
    """
    State-value network for A2C.
    Outputs a scalar value estimate for each state.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class ActorCriticA2C(BaseAlgorithm):
    """
    Advantage Actor-Critic (A2C) algorithm.
    """
    def __init__(
        self,
        device=None,
        num_of_action: int = 2,
        action_range: list = [-2.5, 2.5],
        n_observations: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
    ) -> None:
    
    
        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
        self.device = device or torch.device('cpu')
        self.actor = Actor(n_observations, hidden_dim, num_of_action).to(self.device)
        self.critic = Critic(n_observations, hidden_dim).to(self.device)
        # Combined optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        self.episode_durations = []
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()

    def select_action(self, state: torch.Tensor):
        """Returns action, log_prob, and value estimate for the given state."""
        logits = self.actor(state.unsqueeze(0))
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        value = self.critic(state.unsqueeze(0)).squeeze(0)
        return action, log_prob, value

    def learn(self, env, max_steps: int = 500):
        """
        Run one episode, collect trajectory, and perform A2C updates.
        Returns:
            total_reward (float), episode_length (int)
        """
        # initialize
        obs, _ = env.reset()
        state = to_tensor(obs, self.device)
        log_probs = []
        values = []
        rewards = []
        total_reward = 0.0
        done = False
        t = 0

    
        while not done and t < max_steps:
            a, lp, v = self.select_action(state)
            # turn that int into the proper tensor shape
            action_tensor = self.scale_action(a)        # shape: (1,)
            action_tensor = action_tensor.unsqueeze(0)  # shape: (1,1)
            action_tensor = action_tensor.to(self.device)
            next_obs, r, terminated, truncated, _ = env.step(action_tensor)
            reward = float(r)

            log_probs.append(lp)
            values.append(v)
            rewards.append(reward)
            total_reward += reward

            state = to_tensor(next_obs, self.device)
            done = terminated or truncated
            t += 1

        # compute returns and advantages
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        # advantages
        advantages = returns - values

        # compute losses
        actor_loss = - (log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return total_reward, actor_loss, critic_loss,loss, t
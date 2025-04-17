import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from RL_Algorithm.RL_base_function import BaseAlgorithm

# Helper to convert observations to tensors on the correct device
def to_tensor(obs, device):
    if isinstance(obs, dict):
        return obs['policy'].squeeze().to(device)
    return torch.tensor(obs, dtype=torch.float32, device=device)

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(Actor, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # ====================================== #
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        # ========= put your code here ========= #
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.tanh(self.fc3(x))  # actions in [-1,1]
        # ====================================== #

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, dropout=0.1):
        super(Critic, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        # ====================================== #
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        # ========= put your code here ========= #
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
        # ====================================== #

class Actor_Critic(BaseAlgorithm):
    def __init__(
        self,
        device=None,
        num_of_action: int = 2,
        action_range: list = [-2.5, 2.5],
        n_observations: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.05,
        learning_rate: float = 1e-4,
        tau: float = 0.005,
        discount_factor: float = 0.95,
        buffer_size: int = 256,
        batch_size: int = 32,
    ):
        # Initialize networks
        self.device = device or torch.device('cpu')
        self.actor = Actor(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, dropout).to(self.device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim, dropout).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Hyperparameters
        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor
        self.action_range = action_range

        # Initialize replay buffer inside BaseAlgorithm
        super(Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # Copy weights to target networks
        self.update_target_networks(tau=1.0)

    def select_action(self, state, noise_scale=0.0):
        # ========= put your code here ========= #
        state_t = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0)
        # add exploration noise
        if noise_scale > 0:
            noise = torch.randn_like(action) * noise_scale
            action = action + noise
        # clamp to [-1,1]
        action = action.clamp(-1, 1)
        # scale to environment range
        low, high = self.action_range
        scaled = low + (action + 1.0) * 0.5 * (high - low)
        return scaled, action
        # ====================================== #

    def generate_sample(self):
        # ========= put your code here ========= #
        if len(self.memory) < self.batch_size:
            return None
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions).squeeze(1)
            y = rewards + self.discount_factor * (1 - dones) * target_q
        q = self.critic(states, actions).squeeze(1)
        critic_loss = F.mse_loss(q, y)
        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        return critic_loss, actor_loss

    def update_policy(self):
        sample = self.generate_sample()
        if sample is None:
            return None
        states, actions, rewards, next_states, dones = sample
        # convert to tensors on device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # compute losses
        critic_loss, actor_loss = self.calculate_loss(states, actions, rewards, next_states, dones)
        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return critic_loss.item(), actor_loss.item()

    def update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        # soft update
        for p, p_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)

    def learn(self, env, max_steps=500, num_agents=1, noise_scale=0.1, noise_decay=0.99):
        obs, _ = env.reset()
        state = to_tensor(obs, self.device)
        total_reward = 0.0
        done = False
        t = 0
        while not done and t < max_steps:
            # select and scale action
            scaled_action, raw_action = self.select_action(state, noise_scale)
            next_obs, reward, terminated, truncated, _ = env.step(scaled_action.unsqueeze(0))
            r = float(reward)
            total_reward += r
            done = terminated or truncated
            next_state = to_tensor(next_obs, self.device)
            # store
            self.memory.add(state, raw_action, r, next_state, float(done))
            # update
            self.update_policy()
            self.update_target_networks()
            # decay noise
            noise_scale *= noise_decay
            state = next_state
            t += 1
        return total_reward, t


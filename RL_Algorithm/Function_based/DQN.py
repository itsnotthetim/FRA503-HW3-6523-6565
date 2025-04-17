
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from RL_Algorithm.RL_base_function import BaseAlgorithm

def to_tensor(obs, device):
    if isinstance(obs, dict):
        return obs['policy'].squeeze().to(device)
    return torch.tensor(obs, dtype=torch.float32, device=device)

class DQN_network(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class DQN(BaseAlgorithm):
    def __init__(
            self,
            device=None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            tau: float = 0.005,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
            buffer_size: int = 1000,
            batch_size: int = 32,
    ) -> None:
        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        self.device = device or torch.device("cpu")
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.tau = tau
        self.batch_size = batch_size
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_action)
        with torch.no_grad():
            q_vals = self.policy_net(state.unsqueeze(0))
            return int(q_vals.argmax(dim=1).item())

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_state_values = torch.zeros_like(state_action_values, device=self.device)
        if non_final_next_states.size(0) > 0:
            next_q = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = next_q.max(dim=1)[0].detach()
        expected_q = reward_batch + self.discount_factor * next_state_values
        return F.mse_loss(state_action_values, expected_q)

    def generate_sample(self):
        if len(self.memory) < self.memory.batch_size:
            return None
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()
        non_final_mask = (done_batch == 0)
        non_final_next_states = next_state_batch[non_final_mask]
        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch

    def update_policy(self):
        sample = self.generate_sample()
        if sample is None:
            return None
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decay_epsilon()
        return loss.item()

    def update_target_networks(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self, env, max_steps=500):
        obs, _ = env.reset()
        state = to_tensor(obs, self.device)
        total_reward = 0.0
        done = False
        t = 0
        while not done and t < max_steps:
            a = self.select_action(state)
            action_tensor = self.scale_action(a).unsqueeze(0).to(self.device)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            r = float(reward)
            total_reward += r
            done = terminated or truncated
            next_state = to_tensor(next_obs, self.device)
            self.memory.add(state, torch.tensor(a, device=self.device),
                            torch.tensor(r, device=self.device), next_state, done)
            state = next_state
            self.update_policy()
            self.update_target_networks()
            t += 1
        return total_reward, t
    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #
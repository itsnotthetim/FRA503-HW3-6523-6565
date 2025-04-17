from __future__ import annotations
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display
import wandb
import random

from RL_Algorithm.RL_base_function import BaseAlgorithm
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DQN_network(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class DQN(BaseAlgorithm):
    def __init__(self, device=None, num_of_action=2, action_range=[-2.5, 2.5], n_observations=4,
                 hidden_dim=64, dropout=0.5, learning_rate=0.01, tau=0.005,
                 initial_epsilon=1.0, epsilon_decay=1e-3, final_epsilon=0.001,
                 discount_factor=0.95, buffer_size=1000, batch_size=1):

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

        self.device = device
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.tau = tau
        self.num_of_action = num_of_action
        self.batch_size = batch_size
        self.episode_durations = []
        self.sum_count = 0
        self.reward_sum = 0
        self.is_ipython = 'inline' in plt.get_backend()
        if self.is_ipython:
            display.display(plt.gcf())

    def select_action(self, state):
        state = state.to(self.device).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_action)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values[0].argmax().item()

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        state_batch = state_batch.to(self.device).squeeze(1)              # [B, state_dim]
        action_batch = action_batch.to(self.device).squeeze(1).long()     # [B]
        reward_batch = reward_batch.to(self.device).squeeze(1)            # [B]
        non_final_next_states = non_final_next_states.to(self.device).squeeze(1)  # [~done, state_dim]

        # Q(s, a) from policy network
        q_values = self.policy_net(state_batch)                           # [B, A]
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [B]

        # max_a' Q_target(s', a') from target net
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states.size(0) > 0:
                next_q_values = self.target_net(non_final_next_states)
                max_next_q_values = next_q_values.max(1)[0]
                next_state_values[non_final_mask] = max_next_q_values

        # Bellman target
        expected_state_action_values = reward_batch + self.discount_factor * next_state_values
        expected_state_action_values = expected_state_action_values.detach()

        # DQN loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        return loss


    def generate_sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        # Convert boolean tensor to mask (True if not done)
        done_batch = done_batch.to(self.device).view(-1)  # shape: [batch]
        non_final_mask = (done_batch == 0.0)

        non_final_next_states = next_state_batch[non_final_mask]

        action_batch = action_batch.to(self.device)
        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch

    def update_policy(self):
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return None

        self.optimizer.zero_grad()
        loss = self.calculate_loss(*sample)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_networks(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * policy_param.data)

    def learn(self, env, obs):
        state = torch.as_tensor(obs["policy"], dtype=torch.float32, device=self.device)
        done = False
        timestep = 0
        episode_return = 0.0
        self.decay_epsilon()
        losses = []

        while not done:
            action_idx = self.select_action(state)
            action_tensor = torch.tensor([[action_idx]], dtype=torch.float32, device=self.device)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            done = terminated or truncated
            next_state = torch.as_tensor(next_obs["policy"], dtype=torch.float32, device=self.device)

            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            action_tensor_long = torch.tensor([[action_idx]], dtype=torch.int64, device=self.device)
            done_tensor = torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=self.device)

            self.memory.add(state.unsqueeze(0), action_tensor_long, reward_tensor.unsqueeze(0), next_state.unsqueeze(0), done_tensor)

            loss_val = self.update_policy()
            if loss_val is not None:
                losses.append(loss_val)
            self.update_target_networks()

            state = next_state
            episode_return += reward
            timestep += 1

        self.sum_count += timestep
        self.reward_sum += episode_return
        self.count = timestep
        avg_loss = np.mean(losses) if losses else 0.0

        wandb.log({
            "reward": episode_return,
            "epsilon": self.epsilon,
            "timestep": timestep,
            "avg_loss": avg_loss,
            "count": self.count
        })

        return next_obs, {
            "reward": episode_return,
            "epsilon": self.epsilon,
            "timestep": timestep,
            "avg_loss": avg_loss
        }

    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Result' if show_result else 'Training...')
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

    def save_w_DQN(self, path, filename):
        os.makedirs(path, exist_ok=True)
        weights = self.policy_net.fc_out.weight.detach().cpu().numpy().tolist()
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(weights, f)

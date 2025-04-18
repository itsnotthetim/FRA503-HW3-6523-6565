import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm
import wandb

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc_mean(x)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for Q-value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()


    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state, action):
        """
        Forward pass for Q-value estimation.

        Args:
            state (Tensor): Current state of the environment.
            action (Tensor): Action taken by the agent.

        Returns:
            Tensor: Estimated Q-value.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc_value(x)

class Actor_Critic(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 1,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)

        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor

        self.update_target_networks(tau=1)  # initialize target networks

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        
        # ====================================== #

        super(Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    def convert_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs["policy"]
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device).float().unsqueeze(0)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state, noise=0.1):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
                - clipped_action: The action before scaling but after noise adjustment.
        """
        state = self.convert_obs(state) 
        logits = self.actor(state)
        dist = Normal(logits, noise)
        action = dist.sample()
        action = torch.clamp(action, min=self.action_range[0], max=self.action_range[1])
        return action.squeeze(0).detach().cpu().numpy()
        
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        if len(self.memory) < batch_size:
            return None
        batch = self.memory.sample()
        # ====================================== #
        
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        return (state_batch.to(self.device), action_batch.to(self.device), reward_batch.to(self.device),
                next_state_batch.to(self.device), done_batch.to(self.device))
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # Compute values
        values = self.critic(states, actions).squeeze()
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_values = self.critic_target(next_states, next_actions).squeeze()

        # Compute TD targets
        td_target = rewards + self.discount_factor * next_values * (1 - dones)
        advantages = td_target - values

        # Critic loss (MSE)
        critic_loss = advantages.pow(2).mean()

        # Actor loss (Policy Gradient with Advantage)
        logits = self.actor(states)
        dist = Normal(logits, 1.0)
        log_probs = dist.log_prob(actions).sum(dim=1)
        actor_loss = -(log_probs * advantages.detach()).mean()

        return critic_loss, actor_loss

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return None, None
        states, actions, rewards, next_states, dones = sample
        critic_loss, actor_loss = self.calculate_loss(states, actions, rewards, next_states, dones)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return critic_loss.item(), actor_loss.item()


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        tau = tau or self.tau
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def learn(self, env, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        actor_loss = 0.0
        critic_loss = 0.0

        while not done and step < max_steps:
            action = self.select_action(obs, noise=noise_scale)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).view(1, -1)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)

            next_state_tensor = torch.tensor(next_obs["policy"], dtype=torch.float32, device=self.device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            done_tensor = torch.tensor([terminated or truncated], dtype=torch.float32, device=self.device)

            self.memory.add(
                torch.tensor(obs["policy"], dtype=torch.float32, device=self.device).unsqueeze(0),
                action_tensor,
                reward_tensor,
                next_state_tensor.unsqueeze(0),
                done_tensor
            )

            obs = next_obs
            total_reward += reward
            step += 1

            c_loss, a_loss = self.update_policy()
            if c_loss is not None and a_loss is not None:
                critic_loss += c_loss
                actor_loss += a_loss

            self.update_target_networks()
            noise_scale *= noise_decay

        avg_actor_loss = actor_loss / step if step else 0.0
        avg_critic_loss = critic_loss / step if step else 0.0

        return total_reward.item() if isinstance(total_reward, torch.Tensor) else total_reward, step, {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss
        }
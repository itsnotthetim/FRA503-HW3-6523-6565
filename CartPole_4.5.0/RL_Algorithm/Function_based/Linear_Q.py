from __future__ import annotations
import numpy as np
import torch
import random
import wandb
from RL_Algorithm.RL_base_function import BaseAlgorithm

class Linear_Q(BaseAlgorithm):
    def __init__(
        self,
        num_of_action: int = 7,
        action_range: list = [-2.5, 2.5],
        learning_rate: float = 0.01,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
    ) -> None:

        self.featureVectorList = []

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

    def convert_obs(self, obs):
        obs = torch.tensor([
            obs["policy"][0][0],
            obs["policy"][0][1],
            obs["policy"][0][2],
            obs["policy"][0][3]
        ])
        return obs.detach().cpu().numpy()

    def update(self, obs, action: int, reward: float, next_obs, next_action: int):
        q_current = self.q(obs, action)
        q_next_list = self.q(next_obs)
        td_target = reward + self.discount_factor * np.max(q_next_list)
        td_error = td_target - q_current
        self.w[:, action] += self.lr * td_error * obs

        return td_error

    def select_action(self, obs, epsilon):
        q_list = self.q(obs)
        if random.uniform(0, 1.0) < epsilon:
            return random.choice(range(self.num_of_action))
        else:
            return int(np.argmax(q_list))

    def mapping_action(self, action: int):
        action_transform = -2 if action == 0 else 2
        return torch.tensor([[action_transform]], dtype=torch.float32)

    def learn(self, env, max_steps, episode=0):
        obs, _ = env.reset()
        obs = self.convert_obs(obs)

        cumulative_reward = 0
        cumulative_loss = 0.0 
        step = 0
        done = False

        self.decay_epsilon()

        while not done and step < max_steps:
            action = self.select_action(obs, self.epsilon)
            action_tensor = self.mapping_action(action)

            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            reward_value = reward.item()
            cumulative_reward += reward_value

            next_obs = self.convert_obs(next_obs)
            next_action = self.select_action(next_obs, self.epsilon)

            td_error = self.update(obs, action, reward_value, next_obs, next_action)
            cumulative_loss += td_error**2

            obs = next_obs
            done = terminated or truncated
            step += 1

        avg_loss = cumulative_loss / max(1, step)

        return cumulative_reward, step, avg_loss
from __future__ import annotations
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """        

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    # def update(
    #     self,
    #     obs,
    #     action: int,
    #     reward: float,
    #     next_obs,
    #     next_action: int,
    #     terminated: bool
    # ):
    #     """
    #     Updates the weight vector using the Temporal Difference (TD) error 
    #     in Q-learning with linear function approximation.

    #     Args:
    #         obs (dict): The current state observation, containing feature representations.
    #         action (int): The action taken in the current state.
    #         reward (float): The reward received for taking the action.
    #         next_obs (dict): The next state observation.
    #         next_action (int): The action taken in the next state (used in SARSA).
    #         terminated (bool): Whether the episode has ended.

    #     """
    #     # ========= put your code here ========= #
    #     if isinstance(obs, torch.Tensor):
    #         obs = obs.detach().cpu().numpy()
    #     if isinstance(next_obs, torch.Tensor):
    #         next_obs = next_obs.detach().cpu().numpy()

    #     q_sa = self.q(obs)[action]
    #     next_q = self.q(next_obs)
    #     max_next_q = np.max(next_q) if not terminated else 0.0
    #     td_target = reward + self.discount_factor * max_next_q
    #     td_error = td_target - q_sa

    #     self.w[:, action] += self.lr * td_error * obs

    #     self.training_error.append(td_error)
    #     # ====================================== #
    
    def update(self, obs, action, reward, next_obs, next_action, terminated):
        # 1. If obs is a dict (e.g. {'policy': tensor}), extract and convert
        if isinstance(obs, dict):
            obs = obs["policy"]
        # 2. If it’s a torch.Tensor (GPU or CPU), detach → move → numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        # 3. Otherwise assume it’s array‑like
        else:
            obs = np.array(obs)

        # Same for next_obs
        if isinstance(next_obs, dict):
            next_obs = next_obs["policy"]
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.detach().cpu().numpy()
        else:
            next_obs = np.array(next_obs)

        # Now all obs/next_obs are plain NumPy arrays
        q_sa = self.q(obs)[action]
        next_q = self.q(next_obs)
        max_next_q = 0.0 if terminated else np.max(next_q)
        td_target = reward + self.discount_factor * max_next_q
        td_error  = td_target - q_sa

        # This addition now mixes only NumPy arrays
        self.w[:, action] += self.lr * td_error * obs
        self.training_error.append(td_error)
    
    def to_np(self,x):
        if isinstance(x, dict):
            x = x["policy"]
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)


    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        state = np.array(state)
        state = self.to_np(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_of_action)
        q_vals = self.q(state)
        return int(np.argmax(q_vals))
        # ====================================== #

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        state, _ = env.reset()
        # if isinstance(state, dict):
        #     state = state["policy"].squeeze().cpu().numpy()
        # else:
        #     state = np.array(state)

        state, _ = env.reset()
        # 1) If the env returns a dict of tensors
        if isinstance(state, dict):
            state = state["policy"].squeeze().cpu().numpy()
        # 2) If it returns a torch.Tensor (possibly on GPU)
        elif isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        # 3) Otherwise assume it's NumPy or array-like
        else:
            state = np.array(state)    

        total_reward = 0
        terminated = False
        t = 0

        while not terminated and t < max_steps:
            action = self.select_action(state)
            # scaled_action = self.scale_action(action)
            scaled_action = self.scale_action(action).unsqueeze(0)
            # print(f"[DEBUG] scaled_action shape: {scaled_action.shape}, value: {scaled_action}")


            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            next_state = self.to_np(next_state)
            if isinstance(next_state, dict):
                next_state = next_state["policy"].squeeze().cpu().numpy()
            elif isinstance(next_state, torch.Tensor):
                next_state = next_state.detach().cpu().numpy()
            else:
                next_state = np.array(next_state)

            self.update(state, action, reward, next_state, None, terminated)
            self.decay_epsilon()

            state = next_state
            total_reward += reward
            t += 1

        return total_reward
        # ====================================== #
    




    
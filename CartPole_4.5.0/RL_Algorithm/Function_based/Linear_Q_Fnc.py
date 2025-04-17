from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import torch
import random

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
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        # terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #

    
        learning_rate = 0.01
        # if obs not in self.featureVectorList:
        #     self.featureVectorList.append( [ obs, action ] )

        # featureValue = [ obs, action ] 

        qCurrent = self.q(obs, action)
        qNextList = self.q(next_obs)
        td_error = reward + self.discount_factor * np.max(qNextList) - qCurrent
        self.w[:, action] += ( learning_rate * ( td_error * obs ) )
        
        # ====================================== #
    def convert_obs(self, obs):

        obs =  torch.tensor( [ obs[ 'policy' ][ 0 ][ 0 ], obs[ 'policy' ][ 0 ][ 1 ],obs[ 'policy' ][ 0 ][ 2 ],obs[ 'policy' ][ 0 ][ 3 ] ] )
        obs = obs.numpy()
        return obs
 

    def select_action(self, obs, epsilon):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        
        qNextList = self.q(obs)
        randomProb = random.uniform( 0, 1.0 )
        if epsilon >= randomProb:
            #   Random all action
            exploreChioce = random.choice( range( self.num_of_action ) )
            return  exploreChioce 
        else:
            #   Choose Action index at Max Q value
            exploitChoice = int( np.argmax(qNextList) )
            return  exploitChoice 
        pass
        # ====================================== #

    def mapping_action( self, action ):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n]
            n (int): Number of discrete actions
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        #   Clip action from 0 - 11 to -5 to 5
        if action == 0:
            actionTransform = -2
        else:
            actionTransform = 2

        tensorAction = torch.tensor( [[ actionTransform ]] )

        return tensorAction
    
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
        obs, _ = env.reset()
        done = False
        cumulative_reward = 0
        step = 0
        obs = self.convert_obs(obs)

        while not done or step >= max_steps:
            action = self.select_action(obs, epsilon = self.epsilon )
            actionValue = self.mapping_action(action)
            next_obs, reward, terminated, truncated, _ = env.step( actionValue )
            reward_value = reward.item()
            cumulative_reward += reward_value

            next_obs = self.convert_obs(next_obs) 
            next_action = self.select_action( next_obs , self.epsilon )
            nextActionValue = self.mapping_action( next_action )

            self.update( obs, action, reward_value , next_obs, next_action)
            done = terminated or truncated
            obs = next_obs
            step += 1
        
        return reward_value, step
        # ====================================== #
    




    
import numpy as np
from typing import Tuple
import random

class QLearningOnline:
    
    def __init__(self, learning_rate: float, discount_factor: float, exploration_rate: float, init: str = "zeros") -> None:
        """
        Initializes the Q-Learning Online agent.

        Args:
            learning_rate (float): The rate at which the agent learns.
            discount_factor (float): The factor by which future rewards are discounted.
            exploration_rate (float): The rate of exploration vs exploitation.
            init (str): Method to initialize the Q-table ("zeros" for zero initialization, or "random" for random initialization).
        """
        
        # Initialize parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Initialize Q-table        
        if init == "zeros":
            self.q_table = np.zeros((3, 4, 4), dtype=float)
        elif init == "random":
            self.q_table = np.random.rand(3, 4, 4)
        else:
            raise ValueError("init must be 'zeros' or 'random'")
        
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Select action using ε-greedy:
        using a = arg max_a Q(s, a) with prob (1 - epsilon);.
        """
        
        # decompose state
        r, c = state
        
        # ε-greedy action selection (Probability of choosing random rather than based on Q)
        if random.random() < self.exploration_rate:
            return random.randint(0, 3)  
        return int(np.argmax(self.q_table[r, c, :])) 

    def update_q_value(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int], next_is_terminal: bool) -> None:
        """
        Q-value update:

            Q(s, a) ← Q(s, a) + alpha [ r - Q(s, a) ]  if s' is terminal state, otherwise,

            Q(s, a) ← Q(s, a) + alpha [ r + gamma max_{a'} Q(s', a') - Q(s, a) ].
        """
        
        # decompose states
        r, c = state
        nr, nc = next_state

        # Current Q(s,a)
        q_sa = self.q_table[r, c, action]

        # Compute target
        if next_is_terminal:
            target = reward 
        else:
            max_next = float(np.max(self.q_table[nr, nc, :])) 
            target = reward + self.discount_factor * max_next 

        # Update Q-value
        self.q_table[r, c, action] = q_sa + self.learning_rate * (target - q_sa)
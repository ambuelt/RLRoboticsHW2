################################################################################
# Assignment : HW2                                                             #
# Annika Buelt                                                                 #
# ---------------------------------------------------------------------------  #
#                                                                              #
# 
################################################################################

import numpy as np
import random
import torch
from collections import deque

class ReplayBuffer():
    """
    Buffer stores a batch_size of tuples to train off-policy RL neural network
    """

    def __init__(self, memory_size: int = 5000):
        """
        Initializes the total memory of the replay buffer to be used after each layer
        """

        self.memory = memory_size
        self.buffer = deque(maxlen = self.memory)


    def add_to_buffer(self, state, action, reward, next_state, done):
        """
        Add a tuple to sample to the buffer after making a transition in environment
        
        Args:
            state (Tuple[int, int]): 
            action (int): 
            reward (float): 
            next_state (array[int, int]): 
            done (bool):
        """
        self.buffer.append((state, action, reward, next_state, done))

    
    def sample(self, batch_size: int):
        """
        Sample a random batch of tuples from the buffer memory

        Args:
            batch_size (int): Determines number of tuples to sample.
        
        Returns:
            Tuple[float, int, float, float, ]: The starting position of the agent in the grid.
        
        """

        # Grab a batch_size length of random tuples from the buffer
        batch = random.sample(self.buffer, batch_size)

        # Seperate tuples into seperate lists
        state_list, action_list, reward_list, next_state_list, if_done_list = zip(*batch)

        return (
            torch.FloatTensor(state_list),
            torch.tensor(action_list, dtype=torch.long),
            torch.FloatTensor(reward_list),
            torch.FloatTensor(next_state_list),
            torch.tensor(if_done_list, dtype=torch.long),
        )

    def __len__(self):
        return len(self.buffer)
            

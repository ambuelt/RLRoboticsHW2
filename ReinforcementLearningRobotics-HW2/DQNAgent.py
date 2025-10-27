################################################################################
# Assignment : HW2                                                             #
# Annika Buelt                                                                 #
# ---------------------------------------------------------------------------  #
#                                                                              #
# 
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

import numpy as np
import random
from collections import deque
import copy
from typing import Tuple

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer

BREAK_CON = 0.00008

class DQNAgent():
    def __init__(self, grid,
                 state_dim, action_dim, hidden_layer_size,
                 lr: float, df: float, er: float, tau: float,
                 memory_size: int, batch_size: int,
                 cuda_gpu):
        """
        Constructs the agent to move through the GridWorld Env

        Args:
            grid (GridWorld): The environment to run training
            state_dim (int): Dimension of state space for input into NN (should be 9 or 12)
            action_dim (int): Dimension of the action space used for output of NN (should be 4)
            hidden_layer_size (int): Number of neurons in the hidden layers
            lr (float): The rate at which the agent learns
            df (float): The factor by which future rewards are discounted
            er (float): The rate of exploration vs exploitation
            tau (float): Coeffient for percent to soft update the network to make it so there is no jumping to the target_q_network
            memory_size (int): Size of memory in the ReplayBuffer
            batch_size (int): Size of the sample from the ReplayBuffer to train from
            cuda_gpu (bool): Determines if GPU is used for training
        """

        # Select device to use for training - NEED TO DO
        if (torch.cuda.is_available() and cuda_gpu):
            self.device = "cuda"
        else:
            self.device = "cpu"

        
        # Intialize QNetwork
        self.qnn = QNetwork(state_dim, action_dim, hidden_layer_size).to(self.device)

        # Initialize QNetwork Target with deepcopy to ensure any modifications don't impact self.qnn
        self.target_qnn = copy.deepcopy(self.qnn).to(self.device)

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(memory_size)

        # Select AdamW optimizer
        self.optimizer = optim.AdamW(self.qnn.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()  # Implements Huber Loss

        # Initialize Training Hyperparameters
        self.grid = grid
        self.num_actions = len(grid.actions) - 1   # Set up directional choices: 0=North, 1=East, 2=South, 3=West
        self.discount_factor = df                  # Sets gamma to be used in Bellman equation
        self.exploration_rate = er                 # Determines percent value to select greedy or random action
        self.tau = tau                             # Used for soft updating the q_target
        self.batch_size = batch_size               # Used to determine size of samples to extract

    def encode_state(self, grid, state):
        """
        One-hot encodes the grid position into a flat vector of length grid rows * cols

        Returns:
            A 1D flattened array of 12 values
        """

        r, c = int(state[0]), int(state[1])
        index = r * grid._cols + c

        # Set all non-index values to 0
        one_hot = np.zeros(self.grid._rows * self.grid._cols, dtype=np.float32)

        # Set caclulated index to 1
        one_hot[index] = 1.0

        return one_hot
    
    def choose_action(self, state, er: float) -> int:
        """
        Select action using ε-greedy:
        using a = arg max_a Q(s, a) with prob (1 - epsilon);.
        """
        
        # Convert state tuple to PyTorch Tensor
        # Uses reshape to create new array with 1 row of size equal to original state
        e_state = self.encode_state(self.grid, state)
        tensor_state = torch.FloatTensor(e_state.reshape(1, -1)).to(self.device)

        # Get values from QNetwork and move to CPU to be able to convert to numpy array later
        q_vals_tensor = self.qnn.forward(tensor_state).cpu()

        # Convert to np array and flatten to get 1D array
        with torch.no_grad():
            q_vals = q_vals_tensor.numpy().flatten()
        
        # ε-greedy action selection (Probability of choosing random rather than based on Q)
        if random.random() < er:
            return random.randint(0, self.num_actions)
        
        return int(np.argmax(q_vals))
        
    
    def update_agent(self, grid) -> np.ndarray:
        """
        Trains the QNetwork with batch from replay buffer
    
        Args:
            grid (GridWorld): The environment to run the training
            data_set (list[tuples]): The list of tuples containing (current state, action, reward, and next state) to test
                
        Returns:
            ndarray: A 2D array containing rewards in Qtable
        """

        # Get size of ReplayBuffer
        replay_buffer_size = self.replay_buffer.__len__()
        
        # Check to see if buffer has adequate samples
        # In not, then fill buffer with action
        if (replay_buffer_size < self.batch_size):
            return
        
        # Sample batch of (s, a, r, s', done) tuples from ReplayBuffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        encoded_state = torch.FloatTensor(np.array([self.encode_state(self.grid, s) for s in state])).to(self.device)
        encoded_next_state = torch.FloatTensor(np.array([self.encode_state(self.grid, s) for s in next_state])).to(self.device)
        
        # Compute Current Q Values
        curr_q_vals = self.qnn(encoded_state)           # Computes all Q values for a state
        curr_q = curr_q_vals.gather(1, action.unsqueeze(1)).squeeze(1)  # Grabs the Q value based on the action taken for each state

        # Compute Target Q
        with torch.no_grad():
            target_q_vals = self.target_qnn(encoded_next_state).max(1)[0]

            # Use Q(s, a) ← Q(s, a) + alpha * [ r + gamma max_{a'} Q(s', a') - Q(s, a) ] equation
            # [ r + gamma max_{a'} Q(s', a') - Q(s, a) ]        
            # Compute/Update Target based on best action for the state
            target_q = reward + (self.discount_factor * target_q_vals * (1 - done))

        # Compute Huber Loss using PyTorch functions
        loss = self.loss(curr_q, target_q)

        # Optimize the network
        self.optimizer.zero_grad()  # Sets gradients of all optimized parameters to 0 to prevent previous iterations from adding to current
        loss.backward()             # Starts backpropagation and computes all loss gradients
        self.optimizer.step()       # Updates model parameters based on computed gradients

        self.soft_update_policy(self.tau)
        
        return loss.item()
    

    def soft_update_policy(self, tau: float):
        """
        Updates the policy by a small percentage for each step taken in the env

        Args:
            tau (float): Determines the percentage the target is changed each step
        """
        for param, target_param in zip(self.qnn.parameters(), self.target_qnn.parameters()):
            target_param.data.copy_(tau * param.data + ((1 - tau) * target_param.data))
    
    def evaluate_policy(self, grid):
        """
        Calculates the total reward per optimal policy
    
        Args:
            grid (GridWorld): The environment to run the training
            q_table (ndarray): Holds the updated reward values            
        
        """
        max_steps = 200           # Determines the max steps to take in environment when not converged
        fill_buffer_steps = 64    # Determines how long to run intially to populate buffer for target and q_network comparison
        steps = 0                 # Counts steps walking through greedy policy
        actions = grid.actions    # Set up directional choices: 0=North, 1=East, 2=South, 3=West
        total_reward = 0.0        # Track reward as Qtable is navigated per every test_interval
        state = grid.reset()      # initialize state to starting state
        done = False              # since currently in starting state, not done

        # Run until termincal state (penalty or +1) is reached
        while not done:

            # Don't want to run testing for too long
            if steps >= max_steps:
                break

            best_action = int(self.choose_action(state, 0))

            # ii)  Takes a step in the grid environment
            # Tracks reward and next state
            next_state, reward, done = grid.step(best_action)

            # iii) Sets current state to next state (transistion)
            state = next_state
            total_reward += reward
            steps += 1
        
        return total_reward
    
    def display_policy(self, grid, penalty: int) -> list:
        """
        Creates the dataset for the offline learning algorithm to use to determine best action per state
    
        Args:
            q_table (np.array): 
                
        Returns:
            list: Contains final arrow directional for best action at positions in grid
        """

        # Create dictionary to visually display best policy
        symbol = {0: '↑', 
                  1: '→', 
                  2: '↓', 
                  3: '←'}
        
        symbol_grid = []   # Stores the final arrow, WALL, or value strings for policy display

        # Runs through all grid positions
        for row in range(grid._rows):
            disp_row = []   # Stores every row in the grid (acts as a reset)
            for col in range(grid._cols):

                # Grabs value from current state
                state = (row, col)
                val = grid.grid[state]

                # If state is an obstacle
                if np.isnan(val):
                    disp_row.append('Wall')              # display wall state

                # If state is terminal, stay there
                elif grid.is_terminal(state):
                    if val == grid.penalty:
                        disp_row.append(str(penalty))    # display penalty state
                    else:
                        disp_row.append('+1')            # display final reward state
                
                # If in state where you can move, determine best action (0-3) and add that symbol
                else:
                    best_action_position = int(self.choose_action(state, 0))
                    disp_row.append(symbol[best_action_position])
            
            symbol_grid.append(disp_row)   # add current row to display output

        # Reverse grid so it is display like in word doc position-wise
        symbol_grid.reverse()

        return symbol_grid
    
    def agent_training(self, grid, num_episodes: int, lr: float, df: float, er: float, tau: float, penalty: float, memory_size: int):
        """
        Trains the agent
        """

        training_returns = []
        eval_returns = []
        losses = []
        
        test_interval = 10

        for episode in range(num_episodes):
            max_steps = 100           # Determines the max steps to take in environment when not converged
            fill_buffer_steps = 32    # Determines how long to run intially to populate buffer for target and q_network comparison
            steps = 0                 # Counts steps walking through greedy policy
            actions = grid.actions    # Set up directional choices: 0=North, 1=East, 2=South, 3=West
            total_reward = 0.0        # Track reward as Qtable is navigated per every test_interval
            state = grid.reset()      # initialize state to starting state
            done = False              # since currently in starting state, not done

            steps = 0
            episode_losses = []

            self.exploration_rate = max(0.05, self.exploration_rate * 0.995)

            # Run until termincal state (penalty or +1) is reached
            while not done:

                # Don't want to run testing for too long
                if steps >= max_steps:
                    break
                
                # i) Select best action
                if steps < fill_buffer_steps:
                    action = np.random.randint(0, len(actions) - 1)
                else:
                    action = int(self.choose_action(state, self.exploration_rate))

                # ii)  Takes a step in the grid environment
                # Tracks reward and next state
                next_state, reward, done = grid.step(action)

                # iii) Store action in ReplayBuffer
                self.replay_buffer.add_to_buffer(state, action, reward, next_state, done)

                # Update the agent
                loss = self.update_agent(grid)
                if loss is not None:
                    episode_losses.append(loss)

                # iv) Sets current state to next state (transistion)
                state = next_state
                total_reward += reward
                steps += 1

            training_returns.append(total_reward)
            losses.append(np.mean(episode_losses) if episode_losses else 0)

            #eval_rewards = self.evaluate_policy(grid)
            #eval_returns.append(eval_rewards)
            
            # v) Evaluate the Qlearning
            if (((episode+1) % test_interval) == 0):
                    eval_rewards = self.evaluate_policy(grid)
                    avg_eval = eval_rewards / num_episodes
                    eval_returns.append(avg_eval)

        return training_returns, eval_returns, losses
    
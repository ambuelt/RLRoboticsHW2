import numpy as np
from typing import Tuple

class GridWorld:
    def __init__(self, penalty: float = -1.0) -> None:
        """
        Initializes the GridWorld environment.

        Args:
            penalty (float): The penalty for stepping into the negative terminal state.
            start_state (Tuple[int, int]): The starting position of the agent in the grid.
        """
        
        # Define the grid layout: 0: empty cell, 1: positive terminal state, -1: negative terminal state, np.nan: obstacle
        self.grid = np.array([
            [0,  0,      0,       0],        
            [0,  np.nan, 0,  penalty],       
            [0,  0,      0,       1],        
        ], dtype=float)

        # Grid dimensions and initial state
        self._rows, self._cols = self.grid.shape
        self.start_state = np.array([0, 0])
        self.state = self.start_state

        # Set penalty to be used in displaying policy
        self.penalty = penalty
        
        # Direction relations: 0=up/North,1=right/East,2=down/South,3=left/West
        self.actions = [0, 1, 2, 3]
        
        # cost for each step and slip probability (10% chance to slip left, 10% chance to slip right, 80% chance to go intended direction)
        self.step_cost = -0.04
        self.slip_prob = 0.1
        
        # Direction relations: 0=up,1=right,2=down,3=left
        # maps the intended directions to the directions to slip to
        self._left_slip  = {0: 3, 1: 0, 2: 1, 3: 2}
        self._right_slip= {0: 1, 1: 2, 2: 3, 3: 0}

        self.denorm = False

    def reset(self):
        """
        Resets the environment to the starting state.

        Returns:
            state (Tuple[int, int]): The starting position of the agent in the grid.
        """
        self.state = self.start_state
        normalized_state = self.normalize(self.state)
        return normalized_state

    def is_terminal(self, state) -> bool:
        """
        Checks if the given state is a terminal state.

        Args:
            state (Tuple[int, int]): The state to check.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        state = np.array(state, dtype=int)
        r, c = state

        val = self.grid[r, c]
        return not np.isnan(val) and val != 0.0

    def get_next_state(self, state, action: int) -> np.ndarray:
        """
        Determines the next state given the current state and action.

        Args:
            state (Tuple[int, int]): current state of the agent
            action (int): action taken by the agent (0=up,1=right,2=down,3=left)

        Raises:
            ValueError: If the action is not valid.
            
        Returns:
            Tuple[int, int]: The next state after taking the action.
        """
        
        state = np.array(state, dtype=float)

        # Current position
        r, c = state

        r = int(r)
        c = int(c)
        
        if action == 0:      # up
            nr, nc = r + 1, c
        elif action == 1:    # right
            nr, nc = r, c + 1
        elif action == 2:    # down
            nr, nc = r - 1, c
        elif action == 3:    # left
            nr, nc = r, c - 1
        else:
            raise ValueError("action must be 0,1,2,3")

        # Out of bounds or into wall -> stay put
        if not self.in_bounds(nr, nc) or np.isnan(self.grid[nr, nc]):
            return state
        
        state = np.array([nr, nc])
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Takes a step in the environment based on the given action.

        Args:
            action (int): The action taken by the agent (0=up,1=right,2=down,3=left).
            
        Returns:
            Tuple[Tuple[int, int], float, bool]: A tuple containing the next state, reward, and done status.
        """
        state = np.array(self.state, dtype=float)

        # Current position
        #if (self.is_normalized(state)):
        #    de_normalized_state = self.denormalize(state)
        #    r, c = de_normalized_state
        #    self.denorm = True
        #else:
        #    r, c = state

        r, c = state
        curr_r = int(r)
        curr_c = int(c)
        
        
        # If already terminal, stay there
        if self.is_terminal((curr_r, curr_c)):
            if self.denorm:
                normalized_state = self.normalize((curr_r, curr_c))
                self.denorm = False
            else:
                normalized_state = state
            return normalized_state, float(self.grid[curr_r, curr_c]), True
        
        # Validate action
        if action not in (0, 1, 2, 3):
            raise ValueError("action must be 0,1,2,3")

        # If already terminal, stay there; no extra reward after entry
        if self.is_terminal((curr_r, curr_c)):
            if self.denorm:
                normalized_state = self.normalize((curr_r, curr_c))
                self.denorm = False
            else:
                normalized_state = state
            return normalized_state, 0.0, True

        # Sample slip outcome (80/10/10)
        choices = [self._left_slip[action], action, self._right_slip[action]]
        probs   = [self.slip_prob, 1.0 - 2*self.slip_prob, self.slip_prob]
        actual  = int(np.random.choice(choices, p=probs))

        # Get next state based on actual action taken
        next_state = self.get_next_state(np.array([curr_r, curr_c]), actual)
        
        nr, nc = next_state
        
        # Determine reward and done status
        if self.is_terminal((int(nr), int(nc))):
            reward = float(self.grid[(int(nr), int(nc))])
            done = True
        else:
            reward = self.step_cost
            done = False

        # Update state
        self.state = [int(nr), int(nc)]

        # Normalize the state to [0,1] to help train faster and prevents some states from 
        # dominating learning since the NN will naturally give more weight to larger values
        normalized_next_state = self.normalize((int(nr), int(nc)))

        return normalized_next_state, reward, done
    
    def in_bounds(self, r: int, c: int) -> bool:
        """
        Checks if the given row and column indices are within the grid bounds.

        Args:
            r (int): row index
            c (int): column index

        Returns:
            bool: True if within bounds, False otherwise.
        """
        return 0 <= r < self._rows and 0 <= c < self._cols
    
    def is_normalized(self, state: np.ndarray) -> bool:
        """Check if state is within [0, 1] normalized range."""

        return np.all((state >= 0.0) & (state <= 1.0)) & (self.denorm == False)
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize (r, c) state to [0, 1] range."""

        return np.clip(state / np.array([self._rows - 1, self._cols - 1]), 0.0, 1.0)


    def denormalize(self, state: np.ndarray) -> np.ndarray:
        """Convert normalized [0, 1] state back to discrete grid indices."""
        s = np.clip(state, 0.0, 1.0)

        return np.round(s * np.array([self._rows - 1, self._cols - 1])).astype(int)

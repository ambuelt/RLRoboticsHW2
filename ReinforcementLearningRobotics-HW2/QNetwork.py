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

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layer_size: int = 128):
        """
        Initializes QNetwork to take dicrete actions.

        Args:
           state_dim (int): Determines dimensions of state space for first layer in NN
           
        """

        super(QNetwork, self).__init__()
        self.init_layer = nn.Linear(state_dim, hidden_layer_size)
        self.layer1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, action_dim)

    def forward(self, state):
        """
        Performs forward pass in neural network

        Args:
            state (tensor): Input tensor to represent states in env

        Retunrs:
            q_values (tensor): All possible actions for all states after being passed through NN
        """

        layer_1_pass = funct.relu(self.init_layer(state))
        layer_2_pass = funct.relu(self.layer1(layer_1_pass))
        q_values = self.output_layer(layer_2_pass)

        return q_values
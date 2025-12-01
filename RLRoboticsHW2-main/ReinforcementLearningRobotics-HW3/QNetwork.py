################################################################################
# Assignment : HW3                                                             #
# Annika Buelt                                                                 #
# ---------------------------------------------------------------------------  #
#                                                                              #
# Creates CallBack Function to track losses and rewards
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class PPOCallBack(BaseCallback):
    def __init__(self):
        """
        Initializes QNetwork to take continous actions. Acts like the critic network to estimate the state value function. 
           
        """

        super().__init__()
        self.training_rewards = []    # Tracks rollout mean reward value for training
        self.eval_rewards = []        # Tracks the mean reward value for evaulation
        self.actor_loss = []          # Determines policy losses
        self.critic_loss = []         # Determines value losses
        self.entropy_loss = []        # Determines the entropy of CartPole
        self.policy_stability = []    # Determines if the policy has changed from previous runs (should approach 0)

    def _on_step(self):
        """
        Performs forward pass via step in emvironmnet
        """
        logs = self.model.logger.name_to_value  # Grabs current logging values from model
        #logs = getattr(self.model.logger, 'name_to_value', {}) # Grabs current logging values from model
        print(logs)

        # Will check for all policy (actor), value (critic), and entropy losses within the logs using this keyword
        if 'eval/mean_reward' in logs:
            self.eval_rewards.append(logs['eval/mean_reward'])
        
        if 'rollout/ep_rew_mean' in logs:
            self.training_rewards.append(logs['rollout/ep_rew_mean'])
        
        if 'train/approx_kl' in logs:
            self.policy_stability.append(logs['train/approx_kl'])
        
        if 'train/loss' in logs:
            self.actor_loss.append(logs['train/loss'])

        if 'train/value_loss' in logs:
            self.critic_loss.append(logs['train/value_loss'])

        if 'train/entropy_loss' in logs:
            self.entropy_loss.append(logs['train/entropy_loss'])

        return True  # Indicates that training should continue
    
    def _on_rollout_end(self):
        logs = self.model.logger.name_to_value

        if 'rollout/ep_rew_mean' in logs:
            self.training_rewards.append(logs['rollout/ep_rew_mean'])

    
    def _on_training_end(self):
        logs = self.model.logger.name_to_value

        if 'eval/mean_reward' in logs:
            self.eval_rewards.append(logs['eval/mean_reward'])
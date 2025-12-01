################################################################################
# Assignment : HW3                                                             #
# Annika Buelt                                                                 #
# ---------------------------------------------------------------------------  #
#                                                                              #
# Code solves the CartPole Problem using SB3 library
################################################################################

from QNetwork import PPOCallBack
from ContinuousCartPole import ContinuousCartPole

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor



def make_env(render: bool, seed: int):
    """
    Wraps with SB3 Monitor to track episode rewards
    """
        
    if render:
        render_mode_select = 'human'
    else:
        render_mode_select = None

    #env = gym.make('CartPole-v1', render_mode=render_mode_select) # CartPole Discrete Solving
    env = ContinuousCartPole() # Creates continous control env

    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    env = Monitor(env)

    return env



def evaluate_training(model, seed: int, num_episodes: int):
        """
        Evaluates the trained the agent to see how fast and determines how often to evaluate the policy
        """

        eval_returns = []
        test_interval = 100

        eval_env = make_env(render=False, seed=seed) # Create eval environment using seed=10

        for episode in range(int(num_episodes/test_interval)):
            max_steps = 500           # Determines the max steps to take in environment when not converged
            steps = 0                 # Initialize eval run
            total_reward = 0.0        # Track reward as Qtable is navigated per every test_interval
            done = False              # since currently in starting state, not done
            truncated = False         # Not used in code but need it for padding the model

            oberservations, _ = eval_env.reset(seed=seed)

            # Run until termincal state of balance is reached
            while not (done or truncated):

                # Don't want to run testing for too long
                if steps >= max_steps:
                    break
                
                # i) Select best action
                action, _ = model.predict(oberservations, deterministic=True)

                # ii)  Takes a timestep environment
                # Tracks reward and observations
                oberservations, reward, done, truncated, _ = eval_env.step(action)

                # iv) Add to accumulated rewards
                total_reward += reward
                steps += 1

            eval_returns.append(total_reward)

        eval_env.close() # Close evaluation environment

        # Final metrics output
        window = 10
        print("\n=== Metrics (last {} episodes) ===".format(window))
        print("Eval return  :", [f"{v:.2f}" for v in eval_returns[-window:]])

        return eval_returns

def nn_training(seed: int, num_episodes: int, lr: float, df: float, memory_size: int) -> dict:
    """
    Runs PPO agent for a single seed and tracks metrics
    """
    
    log_dir = "./eval_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    render_selection = False  # Choose between human cartpole display or None

    train_env = make_env(render_selection, seed=seed)
    eval_env = make_env(render=False, seed=10) # Create eval environment using seed=10

    ppo_callback = PPOCallBack()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="{log_dir}/best_model/",
        log_path=log_dir,
        eval_freq=int(num_episodes/100),
        deterministic=True,
        n_eval_episodes=5,
    )

    # Create Model of PPO agent using parameters given in SB3 documentation
    model = PPO(
        'MlpPolicy',                            # Multi-Layer Perceptron (MLP) (feedforward neural network)
        env=train_env,                          # Train PPO agent
        learning_rate=lr,                       # Determines step size of optimization algo
        n_steps=1024,                           # Number of env steps to take each episode
        batch_size=64,                          # Number of samples used for gradient update
        n_epochs=5,                             # Number of times n_steps is iterated over
        gamma=df,                               # High DF to prioritize future rewards (consitantcy)
        clip_range=0.2,                         # Detremines how much change occurs between policies   
        max_grad_norm=0.5,                      # Clips the normalized advatnages
        policy_kwargs=dict(net_arch=[64, 64]),  # Used to define the actor (policy) and critic (value) NN and decided the num of hidden layers
        seed=seed,
        gae_lambda=0.95,                        # Generalized Advatange Estimation to balance bias-varience tradoff
        vf_coef=0.5,                            # Value function coeff to control weights in critic network
        verbose=1                               # Outputs to console
    )

    # Update the agent to track losses
    model.learn(total_timesteps=num_episodes, callback=[ppo_callback, eval_callback])

    eval_returns = eval_env.get_episode_rewards()

    training_returns = train_env.get_episode_rewards() # Grabs training rewards from Monitor

    train_env.close() # Close training environment

    # Final metrics output
    window = 10
    print("\n=== Metrics (last {} episodes) ===".format(window))
    print("Average return Training :", [f"{v:.2f}" for v in training_returns[-window:]])
    print("Average return Eval     :", [f"{v:.2f}" for v in eval_returns[-window:]])
    print("Policy Changes          :", [f"{v:.2f}" for v in ppo_callback.policy_stability[-window:]])
    print("Average actor loss      :", [f"{v:.2f}" for v in ppo_callback.actor_loss[-window:]])
    print("Average critc loss      :", [f"{v:.2f}" for v in ppo_callback.critic_loss[-window:]])
    print("Average entropy loss    :", [f"{v:.2f}" for v in ppo_callback.entropy_loss[-window:]])

    # Create dictionary to use to plot all lr and df values on same plot
    results = {
        'model': model,
        'returns': training_returns,
        'eval_returns': eval_returns,
        'policy': ppo_callback.policy_stability,
        'actor_loss': ppo_callback.actor_loss,
        'critic_loss': ppo_callback.critic_loss,
        'entropy_loss': ppo_callback.entropy_loss,
    }
    
    return results


def plot_metrics(results, variable: str, subtitle: str, seed: int):
    """
    Plots curves for a singular parameter sweep rates

    Args:
            results (list): The environment to run the training
            variable (str): Determines what parameter is being varied per plot
            subtitle (str): Displays the type of QLearning being run
            seed (int): Displays the type of QLearning being run

    """
    
    plt.figure(figsize=(16, 4))
    linestyles = ['-', '--', ':']
    lines = itertools.cycle(linestyles)
    marker_list = ['.', 'x', 's', '^', '+', 'D', 'o']
    markers = itertools.cycle(marker_list)

    # Check if there is no varying hyper parameter
    if variable == None:
        plt.suptitle(f'{subtitle}')
    else:
        plt.suptitle(f'{subtitle} for Varied - {variable.upper()}')

    # Plot Returns
    plt.subplot(1, 4, 1)
    seed = 0
    
    # Plot all Training results from dictionary
    for res in results:
        episodes = range(1, len(res['policy'])+1)
        marker = next(markers)
        line = next(lines)
        
        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Seed' + str(seed)
            seed += 1
            plt.plot(episodes, res['policy'], label=plt_label)
        else:
            plt_label = f'Seed {seed:.1f}: {variable.upper()} = {res[variable]}'
            seed += 1
            plt.plot(episodes, res['policy'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Policy Changes")
    plt.title("Policy Changes vs Episodes")
    plt.legend()
    plt.grid(True)

    # Plot Average change in Q
    plt.subplot(1, 4, 2)
    markers = itertools.cycle(marker_list) # Reset Marker list
    lines = itertools.cycle(linestyles)
    seed = 0

    # Plot all Policy Losses from dictionary
    for res in results:
        episodes = range(1, len(res['actor_loss'])+1)
        marker = next(markers)
        line = next(lines)

        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Seed' + str(seed)
            seed += 1
            plt.plot(episodes, res['actor_loss'], label=plt_label)
        else:
            plt_label = f'Seed {seed:.1f}: {variable.upper()} = {res[variable]}'
            seed += 1
            plt.plot(episodes, res['actor_loss'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Policy Losses")
    plt.title("Policy Loss of PPO")
    plt.legend()
    plt.grid(True)

    # Plot Returns
    plt.subplot(1, 4, 3)
    markers = itertools.cycle(marker_list) # Reset Marker list
    lines = itertools.cycle(linestyles)
    seed = 0
    
    # Plot all Losses from dictionary
    for res in results:
        episodes = range(1, len(res['critic_loss'])+1)
        marker = next(markers)
        line = next(lines)

        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Seed' + str(seed)
            seed += 1
            plt.plot(episodes, res['critic_loss'], label=plt_label)
        else:
            plt_label = f'Seed {seed:.1f}: {variable.upper()} = {res[variable]}'
            seed += 1
            plt.plot(episodes, res['critic_loss'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Value Losses")
    plt.title("Value Losses of PPO")
    plt.legend()
    plt.grid(True)

    # Plot Returns
    plt.subplot(1, 4, 4)
    markers = itertools.cycle(marker_list) # Reset Marker list
    lines = itertools.cycle(linestyles)
    seed = 0
    
    # Plot all Losses from dictionary
    for res in results:
        episodes = range(1, len(res['entropy_loss'])+1)
        marker = next(markers)
        line = next(lines)

        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Seed' + str(seed)
            seed += 1
            plt.plot(episodes, res['entropy_loss'], label=plt_label)
        else:
            plt_label = f'Seed {seed:.1f}: {variable.upper()} = {res[variable]}'
            seed += 1
            plt.plot(episodes, res['entropy_loss'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Entropy Losses")
    plt.title("Entropy Losses of PPO")
    plt.legend()
    plt.grid(True)

    

def main():

    episodes = 50_000       # Number of episodes for dataset creation (used to train off of)  
    lr = 3e-4               # learning rate
    df = 0.95               # discount factor

    ### NN Learning Single Run with Optimal Parameters #############################################

    losses = []
    training_rewards = []
    eval_rewards = []
    training_list_lengths = []
    eval_list_lengths = []
    all_eval_rewards = []
    
    # Time how long it takes to run offline learning
    tik = time.time()  # Start time

    # Train based off of seeds 0, 1, 2
    trials = 3              # Determines number of times to run NN and compare different seeds

    # Run based on number of seeds
    for trial in range(trials):
        seed = trial # 0, 1, 2
        np.random.seed(seed) 
        torch.manual_seed(seed)  # Sets the rand num so that weights are different for each trials
        
        print(f"\n=== Trail {trial} ===")

        # Train PPO agent
        training_results = nn_training(seed, episodes, lr, df, memory_size=1000)
        
        training_rewards.append(training_results['returns'])
        training_list_lengths.append(len(training_results['returns']))
        losses.append(training_results)


        # Evaluate PPO agent
        eval_rewards.append(training_results['eval_returns'])
        eval_list_lengths.append(len(training_results['eval_returns']))
    
        eval_results = evaluate_training(training_results['model'], seed=10, num_episodes=episodes)
        all_eval_rewards.append(eval_results)

        tok = time.time()  # End time

        runtime = tok-tik
        print(f'Runtime: {runtime:.3f} seconds')

        # Plot Losses
        plot_metrics(losses, variable=None, subtitle=f'LR {lr:.2f}, DF {df:.2f}', seed=seed)


    plt.tight_layout()
    plt.show()

    # Plot Training
    # Make sure all lists are the same size as minimum length from 3 seeds
    run_rew = []
    for run in training_rewards:
        run = run[:min(training_list_lengths)]
        run_rew.append(run)

    train_rew = np.array(run_rew)

    plt.figure()
    x_axis = range(1, min(training_list_lengths)+1)

    # Take Mean and Std per result
    train_mean = np.nanmean(train_rew, axis=0)
    train_std = np.nanstd(train_rew, axis=0)

    plt.plot(x_axis, train_mean, label='Eval Curve')
    plt.fill_between(x_axis, (train_mean - train_std), (train_mean + train_std), alpha=0.2)
    plt.title('PPO CartPole Training (Mean Seeds 0, 1, 2)')
    plt.xlabel('Episodes')
    plt.ylabel('Training Returns')
    plt.tight_layout()
    plt.show()

    # Plot Evaulation
    # Make sure all lists are the same size as minimum length from 3 seeds
    run_rew = []
    for run in eval_rewards:
        run = run[:min(eval_list_lengths)]
        run_rew.append(run)

    eval_rew = np.array(run_rew)

    plt.figure()
    x_axis = range(1, min(eval_list_lengths)+1)

    # Take Mean and Std per result
    eval_mean = np.nanmean(eval_rew, axis=0)
    eval_std = np.nanstd(eval_rew, axis=0)

    plt.plot(x_axis, eval_mean, label='Eval Curve')
    plt.fill_between(x_axis, (eval_mean - eval_std), (eval_mean + eval_std), alpha=0.2)
    plt.title('PPO CartPole Evaluation (Seed 10)')
    plt.xlabel('Episodes')
    plt.ylabel('Eval Returns')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

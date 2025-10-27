import Graph.Graph as g
import OnlineLearning.QLearn_Online as ql
import OfflineLearning.QLearn_Offline as ql_off

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from DQNAgent import DQNAgent

import time
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

def nn_training(grid, agent, num_episodes: int, lr: float, df: float, er: float, tau: float, penalty: float, memory_size: int) -> dict:
    """
    Runs offline Q-learning in the GridWorld environment and tracks metrics
    """

    returns, eval_returns, losses = agent.agent_training(grid, num_episodes, lr, df, er, tau, penalty, memory_size)
    

    # Final metrics output
    window = 10
    print("\n=== Metrics (last {} episodes) ===".format(window))
    print("Average return     :", [f"{v:.2f}" for v in returns[-window:]])
    print("Eval return  :", [f"{v:.2f}" for v in eval_returns[-window:]])
    print("Average loss  :", [f"{v:.2f}" for v in losses[-window:]])

    # Show final Q-Action pair for each grid
    offline_best_output = agent.display_policy(grid, penalty)

    # Print by symbolic best policy by row
    print(f'LR {lr:.2f}, DF {df:.2f}, ER {er:.2f}')
    for row in offline_best_output:
        print(row)


    # Create dictionary to use to plot all lr and df values on same plot
    results = {
        'lr': lr,
        'df': df,
        'er': er,
        'penalty': penalty,
        'returns': returns,
        'eval': eval_returns,
        'losses': losses
    }

    return results


def plot_metrics(results, variable: str, subtitle: str, seed: int):
    """
    Plots curves for a singular parameter sweep rates

    Args:
            results (list): The environment to run the training
            variable (str): Determines what parameter is being varied per plot
            subtitle (str): Displays the type of QLearning being run

    """
    
    plt.figure(figsize=(12, 4))
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
    plt.subplot(1, 3, 1)
    
    # Plot all results from dictionary
    for res in results:
        episodes = range(1, len(res['returns'])+1)
        marker = next(markers)
        line = next(lines)
        
        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Seed' + str(seed)
            seed += 1
            plt.plot(episodes, res['returns'], label=plt_label)
        else:
            plt_label = f'Seed {seed:.1f}: {variable.upper()} = {res[variable]}'
            seed += 1
            plt.plot(episodes, res['returns'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Episode Returns")
    plt.legend()
    plt.grid(True)

    # Plot Average change in Q
    plt.subplot(1, 3, 2)
    markers = itertools.cycle(marker_list) # Reset Marker list
    lines = itertools.cycle(linestyles)
    seed = 1

    # Plot all results from dictionary
    for res in results:
        episodes = range(1, len(res['eval'])+1)
        marker = next(markers)
        line = next(lines)
        
        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Seed' + str(seed)
            seed += 1
            plt.plot(episodes, res['eval'], label=plt_label)
        else:
            plt_label = f'Seed {seed:.1f}: {variable.upper()} = {res[variable]}'
            seed += 1
            plt.plot(episodes, res['eval'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Avg Eval Returns")
    plt.title("Episode Eval Returns")
    plt.legend()
    plt.grid(True)

    # Plot Returns
    plt.subplot(1, 3, 3)
    markers = itertools.cycle(marker_list) # Reset Marker list
    lines = itertools.cycle(linestyles)
    seed = 1
    
    # Plot all results from dictionary
    for res in results:
        episodes = range(1, len(res['losses'])+1)
        marker = next(markers)
        line = next(lines)

        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Seed' + str(seed)
            seed += 1
            plt.plot(episodes, res['losses'], label=plt_label)
        else:
            plt_label = f'Seed {seed:.1f}: {variable.upper()} = {res[variable]}'
            seed += 1
            plt.plot(episodes, res['losses'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Losses")
    plt.title("QNetwork Losses")
    plt.legend()
    plt.grid(True)

    

def main():

    er = 0.75               # Acts as initial starting exploration before decaying is applied
    penalty = -1            # Value in penalty state
    episodes = 3000         # Number of episodes for dataset creation (used to train off of)
    trials = 2              # Determines number of times to run NN and compare different seeds
    
    lr = 0.005              # learning rate
    df = 0.9                # discount factor
    tau = 0.15              # Determines how much to change soft updates
    pen = -1                # Value in penalty state
    
    # Test Parametric Sweeps (can turn on/off so it won't run that long)
    if 0:
        
        # Parameters for Parametric Sweeps
        lr_ = [0.1, 0.3, 0.8]
        df_ = [0.3, 0.9,0.99]

        # i) Create grid world environment
        grid = g.GridWorld(penalty = penalty)

        lr_results, df_results = [], []

        # Time how long it takes to run offline learning
        tik = time.time()  # Start time

        for i in lr_:
            agent = DQNAgent(grid, 
                            state_dim = 12,
                            action_dim = 4, 
                            hidden_layer_size = 144, 
                            lr=i, 
                            df=df_[1], 
                            er=er, 
                            tau=tau, 
                            memory_size = 5000, 
                            batch_size = 32, 
                            cuda_gpu = False)

            nn_res = nn_training(grid, agent, episodes, i, df_[1], er, tau, penalty=pen, memory_size=1000)
            lr_results.append(nn_res)

        for j in df_:
            agent = DQNAgent(grid, 
                            state_dim = 12,
                            action_dim = 4, 
                            hidden_layer_size = 144, 
                            lr=lr_[0], 
                            df=j, 
                            er=er, 
                            tau=tau, 
                            memory_size = 5000, 
                            batch_size = 64, 
                            cuda_gpu = False)

            nn_res = nn_training(grid, agent, episodes, lr_[0], j, er, tau, penalty=pen, memory_size=1000)
            df_results.append(nn_res)


        tok = time.time()  # End time

        runtime = tok-tik
        print(f'Runtime: {runtime:.3f} seconds')

        # Plot all results in one figure
        plot_metrics(lr_results, variable='lr', subtitle=f'DF {df:.2f}, Penalty {pen:.2f}', seed=1)

        plt.tight_layout()
        plt.show()

        plot_metrics(df_results, variable='df', subtitle=f'LR {lr:.2f}, Penalty {pen:.2f}', seed=1)

        plt.tight_layout()
        plt.show()


    ### NN Learning Single Run with Optimal Parameters #############################################

    results = []

    # i) Create grid world environment for new penalty
    grid = g.GridWorld(penalty = pen)

    # Time how long it takes to run offline learning
    tik = time.time()  # Start time

    for trial in range(trials):
        seed = trial
        np.random.seed(seed) 
        torch.manual_seed(seed)  # Sets the rand num so that weights are different for each trials
    
        # ii) Create Q-learning agent
        agent = DQNAgent(grid, 
                        state_dim = 12,
                        action_dim = 4, 
                        hidden_layer_size = 144, 
                        lr=lr, 
                        df=df, 
                        er=er, 
                        tau=tau, 
                        memory_size = 5000, 
                        batch_size = 64, 
                        cuda_gpu = False)
        
        print(f"\n=== Trail {trial} ===")
        nn_res = nn_training(grid, agent, episodes, lr, df, er, tau, penalty=pen, memory_size=1000)

        results.append(nn_res)
        tok = time.time()  # End time

        runtime = tok-tik
        print(f'Runtime: {runtime:.3f} seconds')

        plot_metrics(results, variable=None, subtitle=f'LR {lr:.2f}, DF {df:.2f}, ER {er:.2f}, Penalty {pen:.2f}', seed=seed)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

################################################################################
# Assignment : HW3                                                             #
# Annika Buelt                                                                 #
# ---------------------------------------------------------------------------  #
#                                                                              #
# Creates Continous Control Env, Action, and Observations Space                #
################################################################################

# All Functions that have ideas from https://github.com/MattAlanWright/gym-continuous-cartpole/blob/master/gym_continuous_cartpole/envs/continuous_cartpole_env.py
# are clearly marked by "Source: MattAlanWright"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

import numpy as np
import math

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class ContinuousCartPole(CartPoleEnv):
    def __init__(self):
        """
        Initializes QNetwork to take continous actions. Acts like the critic network to estimate the state value function. 

        Main physics functionality was taken from gymnasium.envs.classic_control.cartpole docs and observation space comment below

        ### Observation Space

        The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

        | Num | Observation           | Min                 | Max               |
        |-----|-----------------------|---------------------|-------------------|
        | 0   | Cart Position         | -4.8                | 4.8               |
        | 1   | Cart Velocity         | -Inf                | Inf               |
        | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
        | 3   | Pole Angular Velocity | -Inf                | Inf               |
           
        """

        super().__init__()

        # Physics Parameters "Source: MattAlanWright"
        self.min_action = -1.0
        self.max_action = 1.0
        self.gravity = 9.8        # Force of gravity constant
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)

        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)

        self.force_mag = 50.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        # Make action space continous
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)

        # Make obs space
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.max_force_magnitude = float(self.force_mag) # Use 50 Newtons as max force

        self.state = None
        self.steps_beyond_done = None


    def step(self, action):
        """
        Performs forward pass in neural network. Main physics functionality was taken from gymnasium.envs.classic_control.cartpole docs
        and continous action space creation changes where found in "Source: MattAlanWright"

        Args:
            state (tensor): Input tensor to represent states in env

        Retunrs:
            q_values (tensor): Tuple of 
        """

        # Apply math to CartPole for continous motion
        accel = float(np.clip(action, self.min_action, self.max_action)[0])
        force = accel * self.max_force_magnitude # F = m*a


        # Physics Calculations "Source: MattAlanWright" and gymnasium.envs.classic_control.cartpole docs ##
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Solve physic using Euler's equation
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        
        # Create semi-implicit euler
        else:
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        
        self.state = (x,x_dot,theta,theta_dot)
        
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0

        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0

        else:
            if self.steps_beyond_done == 0:
                print("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, False, {}  # End of Physics Calculations "Source: MattAlanWright" and gymnasium.envs.classic_control.cartpole docs ##
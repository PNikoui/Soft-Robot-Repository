
# Importing simulation
from PlantSim import python_env

# Importing gym environment
import gym
from gym import spaces

# Other imports
import sys
import random
import numpy as np
import time
from math import pi, sqrt, atan2, cos, sin, acos

# Defining helper functions wrt. reward shaping

def wallhit(scan):
    thresh = 0.30
    return min(scan) < thresh

def goalhit(dist):
    thresh = 0.1
    return dist < thresh

def closer(dist1, dist2):
    return dist2 < dist1

def diff(a1, a2):
    """
    Calculates the difference between two angles as well as the sign from a2 to
    a1.
    """
    phi = abs(a1 - a2) % (2 * pi)
    test = ((a1 - a2 >= 0) & (a1 - a2 <= pi)) or ((a1 - a2 <= -pi) & (a1 - a2 >= -2 * pi))

    if test:
        sign = 1
    else:
        sign = -1

    if phi > pi:
        return 2 * pi - phi, sign
    else:
        return phi, sign

class SRobotEnv(gym.Env):
    """
    Custom Environment that follows gym interface and defines the MIT-racearm
    simulation environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, turns):
        super(SRobotEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

        # Defining the maximum distance to goal (used to normalize)
        self.maxgoaldist = 5

        self.turns = turns

        # Spawning the simulation
        self.sim = python_env(turns, 0)

    def observe(self):
        dist = sqrt((self.goal[0] - self.armx)**2 + (self.goal[1] - self.army)**2) # + (self.goal[2] - self.armz)**2)
        angle_to_goal = atan2((self.goal[1] - self.army), (self.goal[0] - self.armx))
#         dot = self.goal.dot(self.arm);
#         lengthA = self.goal.length();
#         lengthB = self.arm.length();
#         angle_to_goal = acos(dot / (lengthA * lengthB) );
        phi, sign = diff(self.armz, angle_to_goal)
        return dist, phi, sign

    def step(self, action):
        '''
        Defining the step function - returning the observation space, reward and
        whether or not the action caused a reset (hitting goal or wall)
        '''

        # Calculating previous distances and angles
        old_dist, old_phi, old_sign = self.observe()
        self.oldx, self.oldy = self.armx, self.army

        self.sim.action(action)

        # Fetching the resulting lidar as well as postion data
#         sensor = self.sim.lidar()
        self.armx, self.army, self.armz, self.theta = self.sim.arm

        # Calculating current distances and angles
        dist, phi, sign = self.observe()

#         # Defining the observation vector
#         self.obs = list(1 - np.array(sensor) / self.maxgoaldist)
#         act = action[0]
#         self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
#         self.obs.extend(act)

      #### CUSTOM observation vector ####
        sensor = self.sim.TrackStar()
        self.obs = list(np.array(sensor) / self.maxgoaldist)
        act = action[0]
        self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
        self.obs.extend(act)

        # Incrementing step counter
        self.current_step += 1

        if wallhit(sensor):
            print('I HIT A WALL!')
            return self.obs, -99, True, {}

        if goalhit(dist):
            print('I HIT THE GOAL!')
            return self.obs, 0, True, {}

        # Defining the reward shaping
        reward = 1

        return np.array(self.obs).reshape(1, -1), reward, False, {}

    def reset(self, seed):
        """
        Resetting the environment for a new run
        """
        random.seed(seed)
        np.random.seed(seed)

        self.sim = python_env(self.turns, seed)

        # Initializing the goal
        self.goal = self.sim.goal[-1]

        self.sim.spawn(0, 0, 0, 0)

        # Initializing step counter
        self.current_step = 0

        # Recieving sensor information
#         sensor1 = self.sim.lidar()
#         print(sensor1,type(sensor1))
        
#         sensor = [0,0,0]
#         print(type(sensor))
        self.armx, self.army, self.armz, self.theta = self.sim.arm

        # Calculating distances and angles for the observation vector
        dist, phi, sign = self.observe()

#         # Defining the observation vector
#         self.obs = list(1 - np.array(sensor) / self.maxgoaldist)
#         act = [0, 0]
#         self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
#         self.obs.extend(act)

         #### CUSTOM observation vector ####
        sensor = self.sim.TrackStar()
        self.obs = list(np.array(sensor) / self.maxgoaldist)
        act = [0, 0, 0]
        self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
        self.obs.extend(act)

        return np.array(self.obs).reshape(1, -1)

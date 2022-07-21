# Imports
import numpy as np
import shapely.geometry as sg
import time
import math
import random
import matplotlib.pyplot as plt

from TargetPathTrack import racetrack

class python_env(object):

    def __init__(self, turns, seed, plot=False):

        random.seed(seed)
        np.random.seed(seed)

        # Defining racetrack
        rt = racetrack(turns, seed)
        self.map, self.goal = rt.generate(plot)

        self.arm = (0, 0, 0) # x,y,theta

        self.angle_inc = (3 / 2) * math.pi / 29

        self.angle_start = -3 / 4 * math.pi
        self.laser_len = 100

        self.dt = 0.1
        self.wheelbase = 0.325

    def spawn(self, x, y, theta):

        self.arm = (x, y, theta)

#     def kinematic(self, velocity, steering_angle):

#         dthetadt = velocity * math.tan(steering_angle) / self.wheelbase

#         theta = self.arm[2] + dthetadt * self.dt

#         if dthetadt == 0:
#             x = self.arm[0] + self.dt * velocity * math.cos(theta)
#             y = self.arm[1] + self.dt * velocity * math.sin(theta)
#         else:
#             x = self.arm[0] + (velocity / dthetadt) * (math.sin(theta) - math.sin(self.arm[2]))
#             y = self.arm[1] + (velocity / dthetadt) * (math.cos(self.arm[2]) - math.cos(theta))

#         return x, y, theta
    
    def kinematic(self, pressure):
        
        # 2nd order transfer function approximation b = 2 a = 5
        x = 2/(pressure[0]**2 + 5*pressure[0] + 5)

        y = 2/(pressure[1]**2 + 5*pressure[1] + 5)
        
        z = 2/(pressure[2]**2 + 5*pressure[2] + 5)

        return x, y, z 

    def action(self, a):

        self.arm = self.kinematic(a[0][0], a[0][1])

    def lidar(self):

        scan = []

        for i in range(30):
            angle = self.angle_start + i * self.angle_inc
            laser = sg.LineString([(self.arm[0], self.arm[1]),
                                   (self.laser_len * math.cos(self.arm[2] + angle) + self.arm[0],
                                   self.laser_len * math.sin(self.arm[2] + angle) + self.arm[1])])

            point_dist = 5

            int = laser.intersection(self.map)

            # Checking for multiline object
            try:
                coords = int.coords
            except NotImplementedError:
                coords = int.geoms[0].coords

            if coords != []:
                point = list(coords)
                d = math.sqrt((self.arm[0] - point[0][0])**2 + (self.arm[1] - point[0][1])**2)
                if d < point_dist:
                    point_dist = d

            scan.append(point_dist)

        return scan

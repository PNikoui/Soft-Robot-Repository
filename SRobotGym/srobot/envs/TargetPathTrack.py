# Imports
import numpy as np
import shapely.geometry as sg
from shapely.geometry import Point
import matplotlib.pyplot as plt
import math
import random
from descartes import PolygonPatch


# Defining racetract class
class racetrack(object):

    def __init__(self, num_observations, num_turns, seed, max_len=30, min_len=1, max_angle= math.pi / 2, min_angle=- math.pi / 2):
        # Setting seed
        np.random.seed(seed)

        self.num_turns = num_turns
        self.num_observations = num_observations
        self.max_len = max_len
        self.min_len = min_len
        self.max_angle = max_angle
        self.min_angle = min_angle

        self.init_point = (0, -0.14*0.5)

    def calc_new_point(self, prev_point, prev_angle):

        # Pick random angle and length
        theta = np.random.uniform(self.min_angle, self.max_angle) + prev_angle
        len = np.random.uniform(self.min_len, self.max_len)

        # Calculating new point from previous point
        x = prev_point[0] + len * math.cos(theta)
        y = prev_point[1] + len * math.sin(theta)

        return (x, y), theta

    def generate(self, plot=False):

        # Initializing list of points
        points = [self.init_point]
        angles = [0]

        # Looping over turns
        for i in range(self.num_turns):
            prev_point = points[-1]
            prev_angle = angles[-1]
            new_point, new_angle = self.calc_new_point(prev_point, prev_angle)
            points.append(new_point)
            angles.append(new_angle)
           

        # Generating the track
        track = sg.LineString(points)
        outer = track.buffer(1.5)
        inner = outer.buffer(-0.5)
        racetrack = outer - inner

        if plot:
            xs = [a[0] for a in points]
            ys = [a[1] for a in points]
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            ax.set_xlim(min(xs)-5,max(xs)+5)
            ax.set_ylim(min(ys)-5,max(ys)+5)
            ax.plot(xs, ys)
            ax.add_patch(PolygonPatch(racetrack, alpha=1, zorder=2))
            plt.show()

        return racetrack, points
    
    def calc_Circle(self, radius,theta):

        # Calculating new point from previous point
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        return (x, y), theta
    
#     def genCircle(self, plot=False):
        
#         # Initializing list of points
#         points = [self.init_point]
#         angles = [0]
#         # Pick random radius and starting point
#         len = np.random.uniform(self.min_len, self.max_len)
#         Start_seed = random.randrange(self.num_observations)
#         angle_i = np.linspace(Start_seed,2*math.pi+Start_seed,self.num_observations)
#         # Looping over turns
#         for i in range(self.num_observations):
#             prev_point = points[-1]
#             prev_angle = angles[-1]
#             new_point, new_angle = self.calc_Circle(len,angle_i[i])
#             points.append(new_point)
#             angles.append(new_angle)
           

#         # Generating the track
#         track = sg.LineString(points)
#         outer = track.buffer(1.5)
#         inner = outer.buffer(-0.5)
#         racetrack = outer - inner

#         if plot:
#             xs = [a[0] for a in points]
#             ys = [a[1] for a in points]
#             fig = plt.figure(figsize=(10, 10))
#             ax = fig.gca()
#             ax.set_xlim(min(xs)-5,max(xs)+5)
#             ax.set_ylim(min(ys)-5,max(ys)+5)
#             ax.plot(xs, ys)
#             ax.plot(xs[0:2],yx[0:2],'r>')
#             ax.add_patch(PolygonPatch(racetrack, alpha=1, zorder=2))
#             plt.show()
    
#         return racetrack, points
        
    def genCircle(self, plot=False):  ## Use a 50mm by 50mm by 50mmm grid 
        
        len = 0.14*0.5
        angle_i = np.linspace(-math.pi/2,math.pi/2,self.num_observations)
#         Start_seed = random.choice(angle_i)
        Goal_seed = random.randrange(10,self.num_observations-10)  ##  Don't select a target to be too close to zero or the max contraction
#         angle_i = 
#         print(Start_seed)
        # Initializing list of points
        points = [self.init_point]
        angles = [0]
        # Pick random radius and starting point
#         len = np.random.uniform(self.min_len, self.max_len) ## Only for Pisa robot
#         Start_seed = random.randrange(self.num_observations)
#         angle_i = np.linspace(Start_seed,2*math.pi+Start_seed,self.num_observations)

        # Looping over turns
        for i in range(self.num_observations):
            prev_point = points[-1]
            prev_angle = angles[-1]
            new_point, new_angle = self.calc_Circle(len,angle_i[i])
            points.append(new_point)
            angles.append(new_angle)
           

        # Generating the track
        track = sg.LineString(points)
        p0 = Point((self.init_point))
        racetrack = p0.buffer(32, cap_style = 3)
#         inner = border.buffer(-50)
#         racetrack = outer - inner


        if plot:
            
            xs = [a[0] for a in points]
            ys = [a[1] for a in points]
#             zs = [a[2] for a in points]
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
#             ax.set_xlim(min(xs)-5,max(xs)+5)
#             ax.set_ylim(min(ys)-5,max(ys)+5)
            ax.plot(xs, ys, linewidth = 2)
            ax.plot(xs[1:3],ys[1:3],'r--')
            ax.add_patch(PolygonPatch(racetrack, alpha=0.05, zorder=2))
            plt.show()
    
        return racetrack, np.asarray(points[Goal_seed]).reshape(-1,1).T
    
    

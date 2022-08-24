# Imports
import numpy as np
import shapely.geometry as sg
import time
import math
import random
import matplotlib.pyplot as plt

from TargetPathTrack import racetrack

class python_env(object):

    def __init__(self, num_observations, turns, seed, Pathtype,plot=False):

        random.seed(seed)
        np.random.seed(seed)

        # Defining racetrack
        rt = racetrack(num_observations,turns, seed)
        if Pathtype == 'Linear':
            self.map, self.goal = rt.generate(plot)
        elif Pathtype == 'Circular':
            self.map, self.goal = rt.genCircle(plot)
        
        self.arm = (0, 0, 0, 0) # x,y,theta

        self.angle_inc = (3 / 2) * math.pi / 29

        self.angle_start = -3 / 4 * math.pi
        self.laser_len = 100

        self.dt = 0.1
        self.wheelbase = 0.325

    def spawn(self, x, y, z, theta):

        self.arm = (x, y, z, theta)

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
    
#     def kinematic(self, pressure):
        
#         # 2nd order transfer function approximation b = 2 a = 5
#         x = self.arm[0] + 1/(pressure[0][0]**2 + 5*pressure[0][0] + 5)

#         y = self.arm[1] + 1/(pressure[0][1]**2 + 5*pressure[0][1] + 5)
        
#         z = self.arm[2] + 1/(pressure[0][2]**2 + 5*pressure[0][2] + 5)
        
#         theta = self.arm[3] + math.atan(x/y)

#         return x, y, z, theta


    def kinematic(self,theta,u):
        
        
        
        F_max = 1000;  #% Maximum force of 1000 N
        eps_max = 0.25; # % Maximum contraction ratio
        a = 0.5;  #% Bicep #(1) active initial length of 50 cm
        b = 0.14*a;
        l0 = a;
        r = 0.025;

        L = sqrt(a^2 + b^2 + 2*a*b*cos(theta)); # % AB distance
        d = a*b*sin(theta)/L; # % E-AB distance
        DL = a + b - L; # % Contracted bicep length variation, DL = L0 - L(theta)
                         #% Triceps elongation is r*theta since r is constant

#         % Actuator (force) static model F(u, eps) = u*F_max*(1 - eps/eps_max) 
#         % u, normalized control variable in (0, 1)
#         % eps, contraction ratio of the muscle (L0 - L(theta))/L0 in (0, eps_max)

#         % => Such a model captures the fundamental variable stiffness spring nature of the
#         %    skeletal muscle without however being able to take into account the passive tension
#         %    peculiar to the skeletal muscle.

#         % => Because the considered muscles are supposed to have the same initial length L0, the
#         %    following static force model results for biceps and triceps:
        eps_b = min(DL/l0, eps_max);
        eps_t = max(eps_max - r*theta/l0, 0);

        F_b = min(u*F_max*(1 - eps_b/eps_max), F_max);
        F_t = min((1 - u)*F_max*(1 - eps_t/eps_max), F_max);

        T_static = d*F_b - r*F_t; # % Equilibium torque

#         % Or, equivalently:
#         % f = F_max*(d*(1 - DL/(l0*eps_max)) + r^2*theta/(l0*eps_max));
#         % g = -F_max*r^2*theta/(l0*eps_max);
#         % 
#         % T_static = u*f + g;

#         T_{dyn} = T_{static} - f_v\dot\theta-MgL_gsin(\theta) - T_{ext}= I\ddot\theta

        T_static - f_v*dtheta - M*g*L_g *sin(theta) - T_ext
        theta = math.asin((I*dd_theta + f_v*d_theta +T_ext)/M*g*L_g)
        return T_static

    def action(self, a):
        
#         print(a)
        
        self.arm = self.kinematic(a)

#         self.arm = self.kinematic(a[0][0], a[0][1],a[0][2])
    
    def lidar(self):

        scan = []

        for i in range(30):
            angle = self.angle_start + i * self.angle_inc
            laser = sg.LineString([(self.arm[0], self.arm[1]),
                                   (self.laser_len * math.cos(self.arm[3] + angle) + self.arm[0],
                                   self.laser_len * math.sin(self.arm[3] + angle) + self.arm[1])])

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
    
    def TrackStar(self): ## Redefine the track to be the whole 2D/3D space, sensor measures to the border and returns wall hit if too close to it
        scan = []

        for i in range(2):
            angle = 0 + i * (math.pi/2)
            laser = sg.LineString([(self.arm[0], self.arm[1]),
                                   (self.laser_len * math.cos(self.arm[3] + angle) + self.arm[0],
                                   self.laser_len * math.sin(self.arm[3] + angle) + self.arm[1])])
            
            
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
     
#         Ex = self.goal[-1][0] - self.arm[0]
#         Ey = self.goal[-1][1] - self.arm[1]
        
#         PathError = [Ex,Ey]
        
#         return PathError
        

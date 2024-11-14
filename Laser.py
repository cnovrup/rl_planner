
import math
import numpy as np
from Obstacle import Obstacle
import pygame
import time

class Laser:
    def __init__(self, range: float, fov: float,laser_readings: int, sample_time: int, log_laser:bool=False) -> None:
        self.range = range
        self.fov = fov
        self.readings = laser_readings
        self.dt = sample_time

        self.angles = [0]*self.readings
        self.ranges = [0]*self.readings
        self.ranges_normalized = [1]*self.readings

        self.angle_min = -(self.fov/2)*(math.pi/180)
        self.angle_max = (self.fov/2)*(math.pi/180)
        self.angle_increment = (self.fov*(math.pi/180))/self.readings
        
        for n,angle in enumerate(self.angles):
            self.angles[n] = (n*self.angle_increment + self.angle_min)

        self.last_sample = time.time()
        self.loglaser = log_laser
        if(log_laser):
            self.logfile = open("laser_log.txt", "w")
    
    def wrap_angle(self, angle):
        wrapped_angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return wrapped_angle

    def get_laser_data(self, obstacles,Ts):
        c = 0
        if(time.time() - self.last_sample > self.dt or self.dt <= Ts):
            start_time = time.time()
            self.ranges = [float('inf')]*self.readings
            self.ranges_normalized = [1]*self.readings
                        
            obstacle:Obstacle
            for i,obstacle in enumerate(obstacles):
                if(obstacle.w == 0 and obstacle.h == 0 and obstacle.radius > 0):
                    obs_radius = float(obstacle.radius)
                    cx = obstacle.x_tractorframe
                    cy = obstacle.y_tractorframe
                    d = math.sqrt(cx**2 + cy**2)
                    phi_c = math.atan2(cy,cx)
                    if(abs(obs_radius/(d)) <= 1):
                        phi_d = math.asin(obs_radius/(d))
                        phi_1 = self.wrap_angle(phi_c + phi_d)
                        phi_2 = self.wrap_angle(phi_c - phi_d) 
                        index_1 = math.floor((phi_1 - self.angle_min)/self.angle_increment)
                        index_2 = math.floor((phi_2 - self.angle_min)/self.angle_increment)
                        if (index_2 > index_1):
                            indexlist = list(range(index_1, self.readings)) + list(range(0,index_2))
                        else:
                            indexlist = range(index_2, index_1)
                        for n in indexlist:
                            c += 1
                            phi = self.angles[n]
                            Rmin = float('inf')
                            #R = cx*math.cos(phi) + cy*math.sin(phi) - (math.sqrt(2)*((cx**2*math.cos(2*phi) - cy**2*math.cos(2*phi) - cx**2 - cy**2 + 2*obs_radius**2 + 2*cx*cy*math.sin(2*phi))/math.cos(phi/2)**4)**(1/2))/4 - (math.sqrt(2)*math.cos(phi)*((cx**2*math.cos(2*phi) - cy**2*math.cos(2*phi) - cx**2 - cy**2 + 2*obs_radius**2 + 2*cx*cy*math.sin(2*phi))/math.cos(phi/2)**4)**(1/2))/4
                            R = cx*math.cos(phi) - (cx**2*math.cos(phi)**2 - cy**2*math.cos(phi)**2 + obs_radius**2 - cx**2 + cx*cy*math.sin(2*phi))**(1/2) + cy*math.sin(phi)
                            #print((cx**2*math.cos(phi)**2 - cy**2*math.cos(phi)**2 + obs_radius**2 - cx**2 + cx*cy*math.sin(2*phi)) )
                            #R = 10
                            if(not isinstance(R,complex) and R.real<Rmin and R.real > 0 and R <= self.range and R.real < self.ranges[n]):
                                Rmin = R.real
                                self.ranges[n] = Rmin 
                                self.ranges_normalized[n] = Rmin/self.range if Rmin != float("inf") else 1.0
                else:
                    angles = np.ravel(np.arctan2(obstacle.vertices_tractorframe[1, :], obstacle.vertices_tractorframe[0, :]))
                    diffs = np.abs(angles[:, np.newaxis] - angles)
                    diffs = np.where(diffs > np.pi, 2 * np.pi - diffs, diffs)
                    max_diff = np.max(diffs)
                    indices = np.where(diffs == max_diff)
                    i, j = indices[0][0], indices[1][0]
                    phi_1 = min(angles[i],angles[j])
                    phi_2 = max(angles[i],angles[j])
                                        
                    index_1 = math.ceil((phi_1 - self.angle_min)/self.angle_increment)
                    index_2 = math.ceil((phi_2 - self.angle_min)/self.angle_increment)
                    if(phi_1 < 0 and phi_2 > 0 and phi_1 < -math.pi/2 and phi_2 > math.pi/2):    
                        #indexlist = range(index_2, index_1)
                        indexlist = list(range(index_2, self.readings)) + list(range(0,index_1))
                    else:
                        indexlist = range(index_1, index_2)
                                            
                    for n in indexlist:
                        R = float('inf')        
                        for i in range(4):
                            if(i<3):
                                x2 = obstacle.vertices_tractorframe[0,i+1]
                                y2 = obstacle.vertices_tractorframe[1,i+1]
                                x1 = obstacle.vertices_tractorframe[0,i]
                                y1 = obstacle.vertices_tractorframe[1,i]
                            else:
                                x2 = obstacle.vertices_tractorframe[0,0]
                                y2 = obstacle.vertices_tractorframe[1,0]
                                x1 = obstacle.vertices_tractorframe[0,3]
                                y1 = obstacle.vertices_tractorframe[1,3]
                            
                            
                            phi = self.angles[n]
                        
                            t = (y1*math.cos(phi) - x1*math.sin(phi))/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
                            r = -(x1*y2 - x2*y1)/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
                            if((t <= 1 and t >= 0) and r < R and r < self.ranges[n]):
                                R = r
                                self.ranges[n] = R
                                self.ranges_normalized[n] = R/self.range if R != float("inf") else 1.0
                                
                        

            '''for n in range(self.readings):
                phi = (n*self.angle_increment + self.angle_min) # calculating angle of laser line
                        
                Rmin = float('inf')
                R = float('inf')
                obstacle:Obstacle
                for i,obstacle in enumerate(obstacles):
                    
                    phi = float(phi)
                    if(obstacle.w > 0 and obstacle.h > 0):
                        angles = np.arctan2(obstacle.vertices_tractorframe[1, :], obstacle.vertices_tractorframe[0, :])
                        min_angle = np.min(angles)
                        max_angle = np.max(angles)
                        if(phi >= min_angle and phi <= max_angle):
                            for i in range(4):
                                if(i<3):
                                    x2 = obstacle.vertices_tractorframe[0,i+1]
                                    y2 = obstacle.vertices_tractorframe[1,i+1]
                                    x1 = obstacle.vertices_tractorframe[0,i]
                                    y1 = obstacle.vertices_tractorframe[1,i]
                                else:
                                    x2 = obstacle.vertices_tractorframe[0,0]
                                    y2 = obstacle.vertices_tractorframe[1,0]
                                    x1 = obstacle.vertices_tractorframe[0,3]
                                    y1 = obstacle.vertices_tractorframe[1,3]

                                t = (y1*math.cos(phi) - x1*math.sin(phi))/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
                                r = -(x1*y2 - x2*y1)/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
                                if((t <= 1 and t >= 0) and r < R):
                                    R = r      
                    else:
                        cx = obstacle.x_tractorframe
                        cy = obstacle.y_tractorframe
                        obs_radius = float(obstacle.radius)
                        d = math.sqrt(cx**2 + cy**2)
                        phi_c = math.atan2(cy,cx)
                        phi_d = math.asin(obs_radius/(d))
                        phi_1 = self.wrap_angle(phi_c + phi_d)
                        phi_2 = self.wrap_angle(phi_c - phi_d)
                        R = cx*math.cos(phi) - (cx**2*math.cos(phi)**2 - cy**2*math.cos(phi)**2 + obs_radius**2 - cx**2 + cx*cy*math.sin(2*phi))**(1/2) + cy*math.sin(phi)
                        
                    if(not isinstance(R,complex) and R.real<Rmin and R.real > 0 and R <= self.range):
                        Rmin = R.real

                self.angles[n] = phi        
                self.ranges[n] = Rmin 
                self.ranges_normalized[n] = Rmin/self.range if Rmin != float("inf") else 1.0'''
                
            self.last_sample = time.time()
            
            #print(self.last_sample - start_time)
            #print(c)
            if(self.loglaser):
                self.logfile.write(str(self.ranges_normalized)[1:-1] + "\n")

    

    def render_laser(self, surface, x_tractor, y_tractor, yaw_tractor,render_scale,screen_w, screen_h):
        for n in range(self.readings):
            dist = self.ranges[n]
            angle = self.angles[n]
            if(dist != float('inf')):
                laser_tractorframe = dist*np.array([[math.cos(angle)],
                                                    [math.sin(angle)]])
                R = np.array([[math.cos(yaw_tractor), -math.sin(yaw_tractor)],
                              [math.sin(yaw_tractor), math.cos(yaw_tractor)]])
                laser_worldframe = R@laser_tractorframe + np.array([[x_tractor], [y_tractor]])
                point_x = laser_worldframe.item((0,0))
                point_y = laser_worldframe.item((1,0))
                pygame.draw.circle(surface,(255,255,255),(point_x*render_scale + screen_w/2,-point_y*render_scale + screen_h/2), 1)


    

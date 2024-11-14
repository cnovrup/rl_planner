import numpy as np
import math 
import pygame

class Obstacle:
    def __init__(self, x, y, radius, h, w, yaw) -> None:
        self.x = x # x world frame
        self.y = y # y world frame
        self.yaw = yaw # yaw world frame
        self.x_tractorframe = None
        self.y_tractorframe = None
        self.yaw_tractorframe = None
        self.radius = radius # radius footprint
        self.h = h # height footprint
        self.w = w # width footprint
        self.vertices = np.array([[w/2, w/2, -w/2, -w/2],
                                  [h/2, -h/2, -h/2, h/2]])

        
        if(self.w > 0 and self.h > 0):
            R = np.matrix([[math.cos(self.yaw), math.sin(self.yaw)],
                           [-math.sin(self.yaw), math.cos(self.yaw)]])
            self.vertices_rotated = R@(self.vertices) + np.array([[self.x],[self.y]])
        
    def set_coord_tractorframe(self, x_tractor, y_tractor, theta_tractor):
        R = np.array([[math.cos(theta_tractor), math.sin(theta_tractor)],
                    [-math.sin(theta_tractor), math.cos(theta_tractor)]])
        t = np.array([[self.x - x_tractor],[self.y - y_tractor]])
        res = R@t
        
        self.x_tractorframe = res.item((0,0))
        self.y_tractorframe = res.item((1,0))
        self.yaw_tractorframe = self.yaw - theta_tractor
        if(self.w > 0 and self.h > 0):
            tractor_pos = np.array([[x_tractor],
                                    [y_tractor]])
            self.vertices_tractorframe = R@(self.vertices_rotated - tractor_pos)

    def draw(self, surface, scale, screen_w, screen_h):
        if(self.radius > 0): #circular obstacle
            pygame.draw.circle(surface,(255,0,0),((self.x*scale)+screen_w/2,-(self.y*scale)+screen_h/2), self.radius*scale)
        else: # rectangular obstacle
            points = [(self.vertices_rotated[0,0]*scale+screen_w/2, self.vertices_rotated[1,0]*scale+screen_h/2),
                      (self.vertices_rotated[0,1]*scale+screen_w/2, self.vertices_rotated[1,1]*scale+screen_h/2),
                      (self.vertices_rotated[0,2]*scale+screen_w/2, self.vertices_rotated[1,2]*scale+screen_h/2),
                      (self.vertices_rotated[0,3]*scale+screen_w/2, self.vertices_rotated[1,3]*scale+screen_h/2)]

            pygame.draw.polygon(surface, (255,0,0), points)
         

    

    

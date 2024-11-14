from Obstacle import Obstacle
import random
import math

class DynamicObstacle(Obstacle):
    def __init__(self, radius, velocity, max_x, min_x, max_y, min_y):
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        
        while(True):
            x = (max_x - min_x)*random.random() + min_x
            y = (max_y - min_y)*random.random() + min_y
            if(math.sqrt(x**2 + y**2) > 6):
                break
        
        
        self.v = velocity
        self.wp_x = (max_x - min_x)*random.random() + min_x
        self.wp_y = (max_y - min_y)*random.random() + min_y
        
        super().__init__(x,y,radius,0,0,0)
        
    def update(self,dt):
        angle  = math.atan2(self.y - self.wp_y, self.x - self.wp_x)
        if(self.x_tractorframe < 6 +self.radius and self.x_tractorframe > -2-self.radius and abs(self.y_tractorframe) < 3+self.radius):
            step = 0
        else:
            step = self.v*dt
        
        self.x = -step*math.cos(angle) + self.x
        self.y = -step*math.sin(angle) + self.y
        
        if(math.sqrt((self.x - self.wp_x)**2 + (self.y-self.wp_y)**2) <= step):
            self.wp_x = (self.max_x - self.min_x)*random.random() + self.min_x
            self.wp_y = (self.max_y - self.min_y)*random.random() + self.min_y

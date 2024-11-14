import gymnasium as gym
import numpy as np
from gymnasium import spaces
#from Tractor import Tractor
from BaseTractor import BaseTractor, TractorSim, TractorGazebo

import math
import pygame
import time
import random
from Obstacle import Obstacle
from DynamicObstacle import DynamicObstacle
from Laser import Laser
import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot as plt
from PIL import Image

class TractorEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, Ts=0.05, render_bool=False, render_mode="tractor"):
        super().__init__()

        laser_readings = 512

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) #velocity and steering angle

        self.observation_space = spaces.Dict({
            "laser": spaces.Box(low=0, high=1, shape=(laser_readings,), dtype=np.float32),
            "data": spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        })
  

        self.tractor = TractorSim(Ts=Ts)
        self.screen = None
        self.render_mode = render_mode
        self.render_bool = render_bool
        self.render_scale = 10
        self.prev_time = time.time()
        self.screen_w = 600
        self.screen_h = 600

        self.obs_test = []
        self.discounted_rew = 0
        self.reward = 0
        self.elapsed_steps = 0
        self.prev_dist_goal = 0
        self.acc_reward = 0
        self.acc_reward_list = []
        self.max_len_ep = 4000 # timeout limit

        #parameters for random path
        self.min_n = 2 # min number of waypoints
        self.max_n = 2 # max number of waypoints
        self.min_r = 10 # min distance between waypoints
        self.max_r = 10 # max distance between waypoints
        self.min_ang = -math.pi/4 # min angle between waypoints
        self.max_ang = math.pi/4 # max angle between waypoints
        self.min_start_ang = -math.pi/2 # min starting angle (angle of first waypoint)
        self.max_start_ang = math.pi/2 # max starting angle (angle of first waypoint)
        
        
        
        self.n_obstacles = 5 # number of obstacles
        self.obs_r_min = 0.5 # min radius of circular obstacles
        self.obs_r_max = 2 # max radius of circular obstacles
        self.obs_v_min = 0.5 # min velocity of dynamic obstacles
        self.obs_v_max = 2 # max velocity of dynamic obstacles
        self.obs_l_min = 0.5 # min side length of rectangular obstacle
        self.obs_l_max = 3 # max side length of rectangular obstacle 
        self.obs_ratio = 0.5 # ratio of static/dynamic obstacles (0: all static, 1: all dynamic)

        self.log_terminated = False
        self.collision = False
        self.goal_reached = False
        self.laser_img = []
        
        self.goal_reached = False
        self.goal_counter = 0
        
        self.prev_vc = 0
        self.prev_dc = 0
        
        self.pos_log = np.zeros((2,1))
        self.test = True
        
    def place_obstacle_on_path(self, rmin=0.5, rmax=3):
        path_len = len(self.tractor.path)
        segment = random.randint(1,path_len)
        r = (rmax-rmin)*random.random()+rmin
        if(segment==1):
            start_coord = [0,0]
            end_coord = self.tractor.path[0]
            t = 0.5*random.random()+0.5
        else:
            start_coord = self.tractor.path[segment-2]
            end_coord = self.tractor.path[segment-1]
            if(segment == path_len):
                t = 0.5*random.random()-0.5
            else:
                t = random.random()
            
        P = np.array(end_coord) - np.array(start_coord)
        obs_coord = P*t + np.array(start_coord)
        x = obs_coord[0]
        y = obs_coord[1]
        self.obs_test.append(Obstacle(x=x,y=y,radius=r,h=0,w=0,yaw=0))
    
    def place_random_obstacles(self, n_obstacles, rmin=0.5, rmax=3, lmin = 0.5, lmax = 3):
        path_array = np.array(self.tractor.path)
        #path_array = np.vstack([path_array, [0,0]])
        end_coord = self.tractor.path[-1]
        end_x = end_coord[0]
        end_y = end_coord[1]
        
        end_angle = math.atan2(end_y,end_x)
        x_proj = math.cos(end_angle)
        y_proj = math.sin(end_angle)
        sign_x = x_proj/abs(x_proj)
        sign_y = y_proj/abs(y_proj)
        if(random.random()<1):
            max_x = max(20*sign_x,0) #np.max(path_array[:,0])
            min_x = min(20*sign_x,6*sign_x) #np.min(path_array[:,0])
            max_y = max(20*sign_y,0) #np.max(path_array[:,1])
            min_y = min(20*sign_y,0) #np.min(path_array[:,1])
        else:
            max_x = np.max(path_array[:,0])-10
            min_x = 10#np.min(path_array[:,0])
            max_y = np.max(path_array[:,1])
            min_y = np.min(path_array[:,1])
            
        #if(random.random() < 0.4):    
        #    max_x = 20
        #    min_x = -0
        #    max_y = 20
        #    min_y = -20
        
        goal = self.tractor.path[-1]

        for i in range(n_obstacles):
            if(random.random() < 0.5):
                while(True):
                    r = (rmax-rmin)*random.random()+rmin
                    x = (max_x - min_x)*random.random() + min_x
                    y = (max_y - min_y)*random.random() + min_y
                    if(math.sqrt(x**2 + y**2) > 6 and math.sqrt((x-goal[0])**2 + (y-goal[1])**2) > 5):
                        break

                self.obs_test.append(Obstacle(x=x,y=y,radius=r,h=0,w=0,yaw=0))
            else:
                while(True):
                    h = (lmax-lmin)*random.random()+lmin
                    w = (lmax-lmin)*random.random()+lmin
                    yaw = 2*math.pi*random.random()-math.pi
                    x = (max_x - min_x)*random.random() + min_x
                    y = (max_y - min_y)*random.random() + min_y
                    if(math.sqrt(x**2 + y**2) > 6 and math.sqrt((x-goal[0])**2 + (y-goal[1])**2) > 5):
                        break
                        
                self.obs_test.append(Obstacle(x=x,y=y,radius=0,h=h,w=w,yaw=yaw))

    def get_reward(self,observation):
        observation = observation["data"]
        reward = -0
        #r_goal/30, th_goal/math.pi, r_next_goal/30, th_next_goal/math.pi, dist_to_path/30, self.v/5, self.d/0.4
        dist_goal = observation[0]*30
        dist_path = observation[4]*30
        th_goal = observation[1]*math.pi
        v = observation[5]*5

        #goal_x = observation[0]*30
        #goal_y = observation[1]*30
        #goal_prev_x = observation[2]*30
        #goal_prev_y = observation[3]*30
    
        #dist = math.sqrt(goal_x**2 + goal_y**2)
        #dist_prev = math.sqrt(goal_prev_x**2 + goal_prev_y**2)

        #ppo3 reward:
        #diff = self.prev_dist_goal - dist_goal
        #self.prev_dist_goal = dist_goal
        #reward += diff/0.1

        '''if(diff > 0):
            reward += 20*diff'''
        if(v > 0):
            reward += 0.07 * v/5*math.cos(th_goal)
        '''elif(v < 0):
            reward += -0.1'''
            
        reward += -0.05 * abs(th_goal)
        #reward += -0.05 * abs(dist_goal)
        '''reward += -0.1 * dist_path
        reward += -0.1 * dist_goal/30'''

        if(dist_goal < self.tractor.goal_tol):
            reward += 1
        #elif(dist_path > 1):
        reward += -0.001*dist_path

        return reward

    def step(self, action):
        t_last_cp = time.time()
        t_start = time.time()
        checkpoints = {}
        
        Ts = 0.05
        
        v = 5*action[0]
        angle = 0.4*action[1]
        
        self.prev_vc = v
        self.prev_dc = angle
        
      
        self.tractor.set_angle(angle)
        self.tractor.set_vel(v)
        self.tractor.update()
        for obstacle in self.obs_test:
            if(type(obstacle) == DynamicObstacle):
                obstacle.update(Ts)
        
        
        '''self.pos_log = np.append(self.pos_log, np.array([[self.tractor.x],[self.tractor.y]]), axis=1)
        if(self.pos_log.shape[1] > 20):
           self.pos_log = np.delete(self.pos_log,0,1)
           
        m = np.mean(self.pos_log, axis=1)
        pos_c = self.pos_log - m[:, np.newaxis]
        r = np.max(np.sqrt(pos_c[1,:]**2 + pos_c[0,:]**2))
        if(r < 0.001):
            print("stuck")'''
        
        terminated = False
        truncated = False
        info = dict()
        observation = dict()
        observation["data"] = self.tractor.get_observation().astype(np.float32)


        #self.obs_test[0].set_coord_tractorframe(self.tractor.x, self.tractor.y, self.tractor.th)
        
        for obstacle in self.obs_test:
            obstacle.set_coord_tractorframe(self.tractor.x, self.tractor.y, self.tractor.th)
        
        self.collision = self.tractor.check_collision(self.obs_test)
        #if(self.tractor.check_collision(self.obs_test) == False):
        #    self.laser.get_laser_data(self.obs_test, Ts)
        self.tractor.get_laser(self.obs_test)    
        observation["laser"] = self.tractor.ranges_normalized
        
        #observation = np.append(observation, np.array(self.elapsed_steps/self.max_len_ep))
        reward = self.get_reward(observation)
        

        #check for termination statement (good)
        terminated = False
        at_final_goal = self.tractor.get_goal_dist() < 1 and self.tractor.path_goal_index == len(self.tractor.path)-1

        if(at_final_goal):
            reward += 5
            #self.tractor.random_path(self.min_n,self.max_n,self.min_r, self.max_r,self.min_ang,self.max_ang)
            self.goal_reached = True
            info["goal_reached"] = self.goal_reached
            terminated = True


        #check for truncation (bad)
        timeout = self.elapsed_steps >= self.max_len_ep
        far_away =  self.tractor.get_distance_to_path() > 30
        collision = self.tractor.check_collision(self.obs_test)
        if(timeout and not terminated):
            reward += -2
            truncated = True
        if(far_away and not terminated):
            reward += -10
            truncated = True
        if(collision and not terminated):
            reward = -10
            terminated = True
            self.collision = True

        checkpoints["rewards"] = time.time() - t_last_cp
        t_last_cp = time.time()

        #print(checkpoints)
        #print(time.time()-t_start)


        

        self.reward = reward
        self.acc_reward += reward
        if(self.render_bool):
            self.acc_reward_list.append(self.acc_reward)

        self.elapsed_steps += 1
        self.log_terminated = terminated

        #observation = observation["data"]
        self.discounted_rew = reward + 0.99*self.discounted_rew
        if(self.render_bool):
            t = time.time()
            self.render()

            
        if(truncated or terminated):
            info["episode_end"] = True    
            info["collision"] = self.collision
            info["goal_reached"] = self.goal_reached
            info["timeout"] = timeout
            info["far_away"] = far_away
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if(self.render_bool == True):
            time.sleep(0)
        self.tractor.reset()
        self.collision = False
        self.goal_reached = False
        self.prev_vc = 0
        self.prev_dc = 0
        
        if(self.test or True):
            self.obs_test = []
            self.test = False    
            self.tractor.random_path(self.min_n,self.max_n,self.min_r, self.max_r,self.min_ang,self.max_ang, self.min_start_ang, self.max_start_ang)
            #self.tractor.interface.publish_path(self.tractor.path)
            #self.tractor.set_path([[10,0],[20,0],[30,0],[40,0],[50,0],[60,0],[70,0],[80,0],[90,0],[100,0],[100,10],[100,20],[100,30],[90,30],[80,30],[70,30],[60,30],[50,30],[40,30],[30,30],[20,30],[10,30],[0,30]])
            #self.tractor.set_path([[10,0],[30,0],[50,0],[80,0],[90,0],[100,0],[100,10],[100,20],[100,30],[90,30],[70,30],[50,30],[30,30],[10,30],[0,30]])
            #self.tractor.set_path([[10,0],[20,0]])
            #self.place_obstacle_on_path(self.obs_r_min, self.obs_r_max)

            '''self.obs_test = [Obstacle(x=66.2598, y=-5.7836, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=68.2638, y=-2.37281, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=73.7663, y=-5.34419, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=73.1971, y=-0.475493, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=68.8046, y=0.476731, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=75.8361, y=2.63417, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=69.4837, y=4.13933, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=72.5457, y=6.01424, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=52.7269, y=26.0629, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=56.9865, y=20.7683, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=59.6295, y=25.2811, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=61.0743, y=35.6658, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=60.7475, y=30.5035, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=56.7869, y=27.2664, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=56.8025, y=33.0797, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=52.0712, y=31.0915, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=52.2602, y=36.8985, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=57.3075, y=37.4889, radius=1.6,h=0,w=0,yaw=0),
                            Obstacle(x=59.2658, y=41.9009, radius=1.6,h=0,w=0,yaw=0)]'''

            for i in range(self.n_obstacles):
                if(random.random() < self.obs_ratio):
                    self.place_random_obstacles(1, self.obs_r_min, self.obs_r_max, self.obs_l_min, self.obs_l_max)
                else:
                    end_coord = self.tractor.path[-1]
                    end_x = end_coord[0]
                    end_y = end_coord[1]
                    
                    end_angle = math.atan2(end_y,end_x)
                    x_proj = math.cos(end_angle)
                    y_proj = math.sin(end_angle)
                    sign_x = x_proj/abs(x_proj)
                    sign_y = y_proj/abs(y_proj)

                    max_x = max(30*sign_x,0) #np.max(path_array[:,0])
                    min_x = min(30*sign_x,6*sign_x) #np.min(path_array[:,0])
                    max_y = max(30*sign_y,0) #np.max(path_array[:,1])
                    min_y = min(30*sign_y,0) #np.min(path_array[:,1])
                    
                    #max_x = 20
                    #min_x = -20
                    #max_y = 20
                    #min_y = -20
                    
                    r = (1-0.5)*random.random()+0.5
                    v = (1.5-0.5)*random.random()+0.5
                    self.obs_test.append(DynamicObstacle(r,v,max_x, min_x, max_y, min_y))
                    if(random.random() < 0.5):
                       self.obs_test.append(DynamicObstacle(r,v,-5, -10, 10,-10))
        else:            
            self.tractor.set_path(self.tractor.path)            
                
        
       

        for obstacle in self.obs_test:
            obstacle.set_coord_tractorframe(self.tractor.x, self.tractor.y, self.tractor.th)
        #self.laser.get_laser_data(self.obs_test, 0.05)
        observation = dict()
        
        observation["data"] = self.tractor.get_observation().astype(np.float32)

        self.tractor.get_laser(self.obs_test)    
        observation["laser"] = self.tractor.ranges_normalized
        #observation = np.append(observation, np.array(self.elapsed_steps/self.max_len_ep))
        self.prev_dist_goal = observation["data"][0]*30
        #print(self.acc_reward)
        

        if(self.render_bool):
           self.render()
           print(self.acc_reward, self.discounted_rew)
           '''if(len(self.acc_reward_list) > 0):
                plt.figure()
                plt.plot(self.acc_reward_list)
                plt.savefig("rewards.png")
                self.acc_reward_list = []

                array = np.array(self.laser_img, dtype=np.float32)
                fig, ax = plt.subplots()
                cax = ax.imshow(array, cmap='viridis', aspect='auto')
                
                cb = fig.colorbar(cax)
                plt.savefig('laser_plot.png', bbox_inches='tight', pad_inches=0)
                plt.axis('off')
                cb.remove()
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig("laser.png", bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                print(array.shape)
                np.save("laser_log", array)
                self.laser_img = []
                plt.close()'''

        #observation = observation["data"]
        self.acc_reward = 0
        info = dict()
        self.discounted_rew = 0
        self.reward = 0
        self.elapsed_steps = 0
        return observation, info

    def distance_to_rectangle_edges(self, phi):
        vertices = np.array([[self.tractor.front, self.tractor.rear, self.tractor.rear, self.tractor.front],
                             [self.tractor.left, self.tractor.left, self.tractor.right, self.tractor.right]])
        R = float('inf')
        for i in range(4):
            if(i<3):
                x2 = vertices[0,i+1]
                y2 = vertices[1,i+1]
                x1 = vertices[0,i]
                y1 = vertices[1,i]
            else:
                x2 = vertices[0,0]
                y2 = vertices[1,0]
                x1 = vertices[0,3]
                y1 = vertices[1,3]

            t = (y1*math.cos(phi) - x1*math.sin(phi))/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
            r = -(x1*y2 - x2*y1)/(y1*math.cos(phi) - y2*math.cos(phi) - x1*math.sin(phi) + x2*math.sin(phi))
            if((t <= 1 and t >= 0) and r < R and r > 0):
                R = r
        return R

    def draw_tractor(self,surface,tractor_coord,angle):

        x = tractor_coord[0]
        y = tractor_coord[1]
        R = np.array([[math.cos(-angle), math.sin(-angle)],
                     [-math.sin(-angle), math.cos(-angle)]])
        vertices = np.array([[self.tractor.front, self.tractor.rear, self.tractor.rear,  self.tractor.front],
                             [self.tractor.left,  self.tractor.left, self.tractor.right, self.tractor.right]])
        
        vertices_rotated = R@vertices + np.array([[x],[y]])
        #vertices_rotated.astype(np.int32)
        
        points = [self.coord_to_px(vertices_rotated[0,0], vertices_rotated[1,0]),
                self.coord_to_px(vertices_rotated[0,1], vertices_rotated[1,1]),
                self.coord_to_px(vertices_rotated[0,2], vertices_rotated[1,2]),
                self.coord_to_px(vertices_rotated[0,3], vertices_rotated[1,3])]

        pygame.draw.polygon(surface, (145,7,7), points)
        #pygame.draw.circle(surface,(255,0,0),((self.tractor.goal_x*self.render_scale)+self.screen_w/2,-(self.tractor.goal_y*self.render_scale)+self.screen_h/2), 5)
    
    def coord_to_px(self, x, y):
        if(self.render_mode == "tractor"):
            rotated_coord = self.tractor.pose_to_tractorframe(x,y,0)
            x = rotated_coord[0]
            y = rotated_coord[1]
            y_px = -x*self.render_scale + self.screen_w/2
            x_px = -y*self.render_scale + self.screen_h/2
        elif(self.render_mode == "world"):
            y_px = -x*self.render_scale + self.screen_w/2
            x_px = -y*self.render_scale + self.screen_h/2

        return (x_px, y_px)
    
    def draw_goal(self, surface, goal_coord, yaw, screen_w, screen_h):
        base_width = 6
        base_length = 12
        tip_width = 12
        tip_length = 6
        x = goal_coord[0]
        y = goal_coord[1]
        vertices = np.array([[-base_length, 0, 0, tip_length, 0, 0, -base_length],
                             [base_width/2, base_width/2, tip_width/2, 0, -tip_width/2, -base_width/2, -base_width/2]])
        R = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
        px_coord = self.coord_to_px(x,y)
        vertices_rotated = R@vertices + np.array([[px_coord[0]],[px_coord[1]]])
        points = []
        for i in range(7):
            x_vertice = vertices_rotated[0,i]
            y_vertice = vertices_rotated[1,i]
            points.append((x_vertice,y_vertice))
        pygame.draw.polygon(surface, (0,0,255), points)
        
    def draw_path(self,surface, screen_w, screen_h, scale):
        for i in range(len(self.tractor.path)):
            if(i == 0):
                x1 = self.tractor.start_x
                y1 = self.tractor.start_y
                x2 = self.tractor.path[i][0]
                y2 = self.tractor.path[i][1]
            else:
                x1 = self.tractor.path[i-1][0]
                y1 = self.tractor.path[i-1][1]
                x2 = self.tractor.path[i][0]
                y2 = self.tractor.path[i][1]

            pygame.draw.line(surface, (0,0,255), self.coord_to_px(x1,y1),self.coord_to_px(x2,y2))

    def draw_projected_point(self,surface, screen_w, screen_h, scale):
        point = self.tractor.get_point_on_path()
        x = point[0]
        y = point[1]
        pygame.draw.circle(surface, (0,0,0),self.coord_to_px(x,y),2)
        pygame.draw.line(surface,(0,0,0), self.coord_to_px(self.tractor.x,self.tractor.y), self.coord_to_px(x,y))

        next_point = self.tractor.get_next_goal()
        next_x = next_point[0]
        next_y = next_point[1]
        point = self.tractor.get_point_on_path(self.tractor.goal_x, self.tractor.goal_y, next_x,next_y)
        x = point[0]
        y = point[1]
        pygame.draw.line(surface,(0,0,0), self.coord_to_px(self.tractor.x,self.tractor.y), self.coord_to_px(x,y))

    def draw_obstacle(self,surface, obstacle:Obstacle):
        if(obstacle.radius > 0): #circular obstacle
            pygame.draw.circle(surface,(255,0,0),self.coord_to_px(obstacle.x, obstacle.y), obstacle.radius*self.render_scale)
        else: # rectangular obstacle
            points = [self.coord_to_px(obstacle.vertices_rotated[0,0], obstacle.vertices_rotated[1,0]),
                      self.coord_to_px(obstacle.vertices_rotated[0,1], obstacle.vertices_rotated[1,1]),
                      self.coord_to_px(obstacle.vertices_rotated[0,2], obstacle.vertices_rotated[1,2]),
                      self.coord_to_px(obstacle.vertices_rotated[0,3], obstacle.vertices_rotated[1,3])]

            pygame.draw.polygon(surface, (255,0,0), points)

    def print_info(self, info):
        label_str = ""
        spacing = 20 #px
        font = pygame.font.Font('freesansbold.ttf', 16)
        for i,key in enumerate(info):
            label_str = key + ": " + str(round(info[key],4))
            text = font.render(label_str, True, (0,0,0), None)
            textRect = text.get_rect()
            textRect.topleft = (10, 10+i*spacing)
            self.surf.blit(text, textRect)
    
    def draw_laser(self, surface, x_tractor, y_tractor, yaw_tractor):
        for n in range(self.tractor.readings):
            dist = 30*self.tractor.ranges_normalized[n]
            angle = n*2*math.pi/self.tractor.readings - math.pi
            if(dist != float('inf')):
                laser_tractorframe = dist*np.array([[math.cos(angle)],
                                                    [math.sin(angle)]])
                R = np.array([[math.cos(yaw_tractor), -math.sin(yaw_tractor)],
                              [math.sin(yaw_tractor), math.cos(yaw_tractor)]])
                laser_worldframe = R@laser_tractorframe + np.array([[x_tractor], [y_tractor]])
                point_x = laser_worldframe.item((0,0))
                point_y = laser_worldframe.item((1,0))
                pygame.draw.circle(surface,(255,255,255),self.coord_to_px(point_x,point_y), 1)
                #pygame.draw.line(surface,(0,0,0),self.coord_to_px(self.tractor.x,self.tractor.y), self.coord_to_px(point_x,point_y))

    def draw_path_points(self):
        if(self.tractor.path != []):
            for coord in self.tractor.path:
                x = coord[0]
                y = coord[1]
                if(x == self.tractor.goal_x and y == self.tractor.goal_y):
                    color = (255,255,255)
                else:
                    color = (0,0,255)

                pygame.draw.circle(self.surf, color, self.coord_to_px(x,y), self.tractor.goal_tol*self.render_scale)

    def draw_lines_wp(self):
        next_goal = self.tractor.get_next_goal()
        pygame.draw.line(self.surf,(0,0,0), self.coord_to_px(self.tractor.goal_x,self.tractor.goal_y), self.coord_to_px(self.tractor.x, self.tractor.y))
        pygame.draw.line(self.surf,(0,0,0), self.coord_to_px(next_goal[0],next_goal[1]), self.coord_to_px(self.tractor.x, self.tractor.y))

    def render(self):
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_w,self.screen_w))

        self.surf = pygame.Surface((self.screen_w,self.screen_w))
        bg_color = (44,189,67)
        self.surf.fill(bg_color)

        self.draw_tractor(self.surf,(self.tractor.x,self.tractor.y),self.tractor.th)
        for obstacle in self.obs_test:
            self.draw_obstacle(self.surf, obstacle)
        
        self.draw_path_points()
        self.draw_laser(self.surf, self.tractor.x, self.tractor.y, self.tractor.th)
        #self.draw_goal(self.surf, (self.tractor.goal_x,self.tractor.goal_y),0, self.screen_w, self.screen_h)
        self.draw_path(self.surf, self.screen_w, self.screen_h,self.render_scale)
        #self.draw_projected_point(self.surf, self.screen_w, self.screen_h,self.render_scale)
        #self.draw_lines_wp()
        
        info_dict = {"x":self.tractor.x,
                     "y":self.tractor.y,
                     "hdg":self.tractor.th,
                     "v":self.tractor.v,
                     "vc":self.tractor.v_command,
                     "d":self.tractor.d,
                     "dc":self.tractor.d_command,
                     "rew":self.reward,
                     "drew":self.discounted_rew,
                     "arew":self.acc_reward,
                     "coll":self.collision}
        
        
        
        self.print_info(info_dict)
        
        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()


    def close(self):
        self.tractor.set_angle(0)
        self.tractor.set_vel(0)
        self.tractor.update()

if __name__ == "__main__":
    a = np.array([0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    a[0] = 1
                if event.key == pygame.K_DOWN:
                    a[0] = -1
                if event.key == pygame.K_LEFT:
                    a[1] = 1
                if event.key == pygame.K_RIGHT:
                    a[1] = -1  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[1] = 0
                if event.key == pygame.K_RIGHT:
                    a[1] = 0
                if event.key == pygame.K_UP:
                    a[0] = 0
                if event.key == pygame.K_DOWN:
                    a[0] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = TractorEnv()
    env.render_bool = True

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            '''if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")'''
            steps += 1
            if terminated or truncated or restart or quit:
                print(steps, total_reward, env.discounted_rew)
                break


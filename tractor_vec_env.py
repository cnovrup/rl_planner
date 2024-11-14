import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Tractor import Tractor
import math
import pygame
import time
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

class TractorEnv(VectorEnv):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, num_envs:int=1):
        super().__init__()

        self.num_envs = num_envs
        self.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.single_observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.tractors:TractorEnv = []
        for i in range(num_envs):
            self.tractors.append(Tractor())

        self.screen = None
        self.render_bool = True
        self.render_scale = 10
        self.prev_time = time.time()
        self.screen_w = 500
        self.screen_h = 500
    

    def step(self, action):
        observation = np.zeros((self.num_envs,6))
        reward = np.zeros((self.num_envs,1))
        terminated = np.zeros((self.num_envs,1))
        for i,tractor in enumerate(self.tractors):
            v = 5*action[i][0]
            angle = 0.4*action[i][0]
            tractor.set_angle(angle)
            tractor.set_vel(v)
            tractor.update(0.001)
            observation[i,:] = tractor.get_norm_observation
            if(self.tractor.get_goal_dist() < 1):
                reward[i] = 100
                terminated[i] = True
            else:
                reward[i] = -self.tractor.get_normalized_dist(math.sqrt(2*50**2))


        '''v = 5*action[0]
        angle = 0.4*action[1]
        self.tractor.set_angle(angle)
        self.tractor.set_vel(v)
        self.tractor.update(0.001)

        terminated = False
        truncated = False
        info = dict()
        observation = self.tractor.get_norm_observation()
        if(self.tractor.get_goal_dist() < 1):
            reward = 100
            terminated = True
        else:
            reward = -self.tractor.get_normalized_dist(math.sqrt(2*50**2)) 
        
        if(self.render_bool):
            self.render()
        #print(self.tractor.x, self.tractor.y, sep="\t")'''
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.tractor.reset()
        #self.tractor.set_random_goal()
        self.tractor.set_goal(10,-5)
        info = dict()
        observation = self.tractor.get_norm_observation()
        print(observation)
        if(self.render_bool):
            self.render()
        return observation, info

    def draw_tractor(self,surface,x,y,angle):
        w = 10
        h = 5
        R = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
        vertices = np.array([[w, -w, -w, w],
                            [h, h, -h, -h]])
        vertices_rotated = R@vertices + np.array([[x],[y]])
        #vertices_rotated.astype(np.int32)

        points = [(vertices_rotated[0,0], vertices_rotated[1,0]),
                (vertices_rotated[0,1], vertices_rotated[1,1]),
                (vertices_rotated[0,2], vertices_rotated[1,2]),
                (vertices_rotated[0,3], vertices_rotated[1,3])]

        pygame.draw.polygon(surface, (255,0,0), points)
        pygame.draw.circle(surface,(255,0,0),((self.tractor.goal_x*self.render_scale)+self.screen_w/2,-(self.tractor.goal_y*self.render_scale)+self.screen_h/2), 5)
        

    def render(self):
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_w,self.screen_w))

        self.surf = pygame.Surface((self.screen_w,self.screen_w))
        bg_color = (44,189,67)
        self.surf.fill(bg_color)
        
        self.draw_tractor(self.surf,(self.tractor.x*self.render_scale)+self.screen_w/2,(-self.tractor.y*self.render_scale)+self.screen_h/2,self.tractor.th)
        font = pygame.font.Font('freesansbold.ttf', 16)
        info_str = str(round(self.tractor.x,2)) + " " + str(round(self.tractor.y,2)) + " " + str(round(self.tractor.th,2)) + " " + str(round(self.tractor.v,2)) + " " + str(round(self.tractor.d,2)) + " " + str(round(self.tractor.get_normalized_dist(math.sqrt(2*50**2)),2)) + " " + str(round( abs(self.tractor.get_heading_error())/math.pi,2))
        text = font.render(info_str, True, (0,0,0), None)
        #text = pygame.transform.flip(text, True, False)
        textRect = text.get_rect()
        textRect.center = (self.screen_w/2, 30)
        self.surf.blit(text, textRect)
        #self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()


    def close(self):
        ...

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
                print(steps)
                break

import numpy as np
from gymnasium.spaces import Dict, Box
import gymnasium as gym
from typing import Any

class DictFrameStack(gym.Wrapper):
    def __init__(self, env, n_stack):
        super(DictFrameStack, self).__init__(env)
        self.n_stack = n_stack
        self.frames = {key: [] for key in env.observation_space.spaces}
        
        # Update observation space
        new_spaces = {key: Box(low=np.min(space.low), high=np.max(space.high), shape=(n_stack,) + space.shape, dtype=space.dtype)
                      for key, space in env.observation_space.spaces.items()}
        self.observation_space = Dict(new_spaces)
    
    def reset(self,seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset()
        for key in self.frames:
            self.frames[key] = [obs[key]] * self.n_stack
        observation = self._get_observation()    
        return observation, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in self.frames:
            self.frames[key].append(obs[key])
            if len(self.frames[key]) > self.n_stack:
                self.frames[key].pop(0)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        return {key: np.array(self.frames[key]) for key in self.frames}

import gymnasium as gym
from tractor_env import TractorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch as th
from CustomFeatureExtractor import CustomCombinedExtractor
from DictFrameStack import DictFrameStack
import time

simulate_real_time = False # True: wait Ts seconds between render update, False: update render as soon as possible
render = True # whether to render graphical interface or not
Ts = 0.05 # simulation step size/sample time in seconds
render_mode = "world" # "tractor": render with respect to tractor reference frame, "world": render wrt. world frame


env = TractorEnv(Ts=Ts, render_bool=render, render_mode=render_mode)
env = DictFrameStack(env,5)
env = gym.wrappers.ClipAction(env)
env = Monitor(env, filename="log")
observation, info = env.reset()

if(simulate_real_time):
    Ts_sim = Ts
else:
    Ts_sim = 0

policy_kwargs = dict(activation_fn=th.nn.Tanh,
                    net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),
                    features_extractor_class=CustomCombinedExtractor
                )


model = PPO.load("models/latest_model.zip",env=env,device="cuda")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()

t = time.time()
goal_reached = False
while(goal_reached == False):
    if(time.time()-t > Ts_sim):
        print(time.time() - t)
        t = time.time()
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)

env.close()

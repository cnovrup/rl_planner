import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from tractor_env import TractorEnv
import torch as th
from SaveBest import SaveBest
from DictFrameStack import DictFrameStack
from CustomFeatureExtractor import CustomCombinedExtractor
import argparse
import os

num_cpu = 1
Ts = 0.05
render = False



def make_env(env_id):
    def _init():
        env = TractorEnv(Ts=Ts, render_bool=render)
        env = DictFrameStack(env,5)
        env = gym.wrappers.ClipAction(env)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", nargs='?', const="ppo", type=str, default="ppo")
    parser.add_argument("--number")
    parser.add_argument("--layer_size", nargs='?', const=256, type=int, default=256)
    parser.add_argument("--n_layers", nargs='?', const=2, type=int, default=2)
    parser.add_argument("--gamma", nargs='?', const=0.99, type=float, default=0.99)
    parser.add_argument("--lr", nargs='?', const= 0.00001, type=float, default=0.00001)
    parser.add_argument("--n_steps", nargs='?', const=1024, type=int, default=1024)
    parser.add_argument("--batch_size", nargs='?', const=64, type=int, default=64)
    parser.add_argument("--ent_coef", nargs='?', const=0.01, type=float, default=0.01)
    parser.add_argument("--sde", nargs='?', const="False", type=str, default="False")
    parser.add_argument("--continue_checkpoint", nargs='?', const=-1, type=int, default=-1)
    parser.add_argument("--dir", nargs='?', const="", type=str, default="")
    parser.add_argument("--seed", nargs='?', const="", type=int, default=0)
    parser.add_argument("--num_cpu", nargs='?', const="", type=int, default=1)
    parser.add_argument("--timesteps", nargs='?', const="", type=int, default=10e6)
    
    args = parser.parse_args()
    alg = args.algorithm
    number = str(args.number)
    layer_size = int(args.layer_size)
    n_layers = int(args.n_layers)
    gamma = float(args.gamma)
    lr = float(args.lr)
    n_steps = int(args.n_steps)
    batch_size = int(args.batch_size)
    ent_coef = float(args.ent_coef)
    seed = int(args.seed)
    timesteps = int(args.timesteps)
    num_cpu = int(args.num_cpu)
    
    sde = False
    if(type(args.sde) == str):
        if(args.sde.lower() == "true"):
            sde = True
    elif(type(args.sde) == int):
        sde = bool(args.sde)
	
    continue_checkpoint = int(args.continue_checkpoint)
    os.makedirs("log", exist_ok=True)
   
    env_id = "Tractor-v1"
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    vec_env = VecMonitor(vec_env, filename="log/monitorlog", info_keywords=("collision","timeout","goal_reached","far_away"))
    
    network_params = [layer_size]*n_layers
    
    '''params = dict()'''
    #model = PPO.load(alg + "_log/latest_model.zip", env=vec_env, device="cuda")
    params = dict()
    
    with open("log/params.txt", "w") as f:
        f.write("algorithm: " + str(alg) + "\n")
        f.write("layer_size: " + str(layer_size) + "\n")
        f.write("n_layers: " + str(n_layers) + "\n")
        f.write("gamma: " + str(gamma) + "\n")
        f.write("learning_rate: " + str(lr) + "\n")
        f.write("n_steps: " + str(n_steps) + "\n")
        f.write("batch_size: " + str(batch_size) + "\n")
        f.write("ent_coef: " + str(ent_coef) + "\n")
        f.write("sde: " + str(sde) + "\n")
        f.write("seed: " + str(seed) + "\n")
        
    #shutil.copyfile("tractor_env.py", alg + "_log" + str(number) + "/tractor_env.py")
    
    if(alg == "ppo"):
        if(continue_checkpoint == -1):
            policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=dict(pi=network_params, vf=network_params), features_extractor_class=CustomCombinedExtractor)
            model = PPO("MultiInputPolicy",vec_env,verbose=1,gamma=gamma,learning_rate=lr,n_steps=n_steps, batch_size=batch_size,tensorboard_log="tb_log/",device="cuda", ent_coef=ent_coef, policy_kwargs=policy_kwargs, use_sde=sde, seed=seed)
        else:
            params = dict()
            cp_path = "checkpoint.zip"
            print("continuing from " + cp_path)
            model = PPO.load(cp_path, env=vec_env, device="cuda", custom_objects=params)
        
   
    callback = SaveBest(check_freq=1000, log_dir="log/", verbose=1)
    model.learn(total_timesteps=timesteps, progress_bar=1, tb_log_name=alg, callback=callback)
    model.save(alg + number)

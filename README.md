# rl_planner
Reinforcement learning for local planning, pathfollowing and obstacle avoidance of a tractor. Made as my masters thesis project for electrical engineering at DTU. This version contains a simplified kinematic model of a tractor with slow driving and steering dynamics and the objective is to follow a path made by waypoints and avoid dynamic and static obstacles. 

## Running the simulator
to run the simulator with a pretrained agent, run `run.py` and the simulator should open, running a pretrained model. If the script `tractor_env.py` is run, the same environment will run but using arrow keys for control. The scripts can be modified to have different sample rate, reference frame, etc.

## training
run '''train.py'''. it accepts a number of arguments which can be found in the file. It runs training with the hyperparameters that were found to work the best. As default it will train from scratch, but to train on a previously trained model add `--continue_checkpoint 1` and it will resume training from `checkpoint.zip` which should be located in the root folder.

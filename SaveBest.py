from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy, load_results
import rospy
from std_srvs.srv import Empty

class SaveBest(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = 1_000_000
        self.log_dir = log_dir
        self.save_path_best = os.path.join(log_dir, "best_model")
        self.save_path_latest = os.path.join(log_dir, "latest_model")
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        self.best_mean_reward = -np.inf

        #rospy.wait_for_service("/gazebo/unpause_physics")
        #rospy.wait_for_service("/gazebo/pause_physics")
        #self.unpause_gazebo = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        #self.pause_gazebo = rospy.ServiceProxy("/gazebo/pause_physics", Empty)


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path_best is not None:
            os.makedirs(self.save_path_best, exist_ok=True)
        if self.save_path_latest is not None:
            os.makedirs(self.save_path_latest, exist_ok=True)
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        '''infos = self.locals['infos']
        for info in infos:
            if("episode_end" in info):
                print(info["collision"])
                print(info["timeout"])
                print(info["far_away"])
                print(info["goal_reached"])'''
        if self.num_timesteps % self.save_freq == 0:
            print("save checkpoint")
            self.save_path_cp = os.path.join(self.checkpoint_dir, "model_" + str(int(self.num_timesteps/1_000_000)) + "M_steps")
            self.model.save(self.save_path_cp)
            
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              self.model.save(self.save_path_latest)
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path_best}")
                  self.model.save(self.save_path_best)

        return True

    '''def _on_rollout_start(self) -> None:
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause_gazebo()

    def _on_rollout_end(self) -> None:
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause_gazebo()'''

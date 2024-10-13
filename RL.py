import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import configure
from utils import CustomActorCriticPolicy, HistoryWrapper


def RL(train: int = 0, model_path: str = None, out: str = None):

    envname = 'LunarLander-v2'
    contextlen = 500
    env = gym.make(envname)
    env = HistoryWrapper(env, history_length=contextlen)

    log_dir = "/tmp/sb3_log/"
    loss_logger = configure(log_dir, ["stdout", "csv"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_steps = env.spec.max_episode_steps


    env = make_vec_env(envname, n_envs=os.cpu_count(), wrapper_class=HistoryWrapper, wrapper_kwargs={'history_length': contextlen})

    model = PPO(CustomActorCriticPolicy, env, verbose=1, policy_kwargs={'contextlen': contextlen})

    if model_path:
        try:
            model = PPO.load(model_path, env, policy_kwargs={'contextlen': contextlen})
        except FileNotFoundError:
            print(f"Model not found at {model_path}")
            return

    model.set_logger(loss_logger)

    if train > 0:
        model.learn(train, progress_bar=True)

    if out:
        model.save(out)

    env.close()


    if train > 0:
        log_file = f"{log_dir}/progress.csv"
        data = pd.read_csv(log_file)

        plt.plot(data['time/total_timesteps'], data['train/loss'])
        plt.xlabel('Timesteps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.savefig('loss.png')
    else:
        vec_env = gym.make(envname, render_mode='human')
        vec_env = HistoryWrapper(vec_env, history_length=contextlen)

        obs, info = vec_env.reset()
        for i in tqdm(range(1000)):
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = vec_env.step(action)
            if terminated or truncated:
                obs, info = vec_env.reset()
            vec_env.render()

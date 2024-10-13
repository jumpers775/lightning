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


def main():

    envname = 'LunarLander-v2'
    contextlen = 500
    env = gym.make(envname)
    env = HistoryWrapper(env, history_length=contextlen)

    log_dir = "/tmp/sb3_log/"
    loss_logger = configure(log_dir, ["stdout", "csv"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_steps = env.spec.max_episode_steps

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    env = make_vec_env(envname, n_envs=os.cpu_count(), wrapper_class=HistoryWrapper, wrapper_kwargs={'history_length': contextlen})

    model = PPO(CustomActorCriticPolicy, env, verbose=1, policy_kwargs={'contextlen': contextlen, "device": device})

    model.set_logger(loss_logger)

    model.learn(1000000 * 7, progress_bar=True)

    env.close()

    log_file = f"{log_dir}/progress.csv"
    data = pd.read_csv(log_file)

    # Plot the loss values
    plt.plot(data['time/total_timesteps'], data['train/loss'])
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.savefig('loss.png')

    # visualize the trained model
    #
    # vec_env = gym.make(envname, render_mode='human')
    # vec_env = HistoryWrapper(vec_env, history_length=contextlen)

    # obs, info = vec_env.reset()
    # for i in tqdm(range(1000)):
    #     action, _states = model.predict(obs)
    #     obs, rewards, terminated, truncated, info = vec_env.step(action)
    #     vec_env.render()


if __name__ == "__main__":
    main()

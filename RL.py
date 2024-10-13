import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from utils import CustomActorCriticPolicy, HistoryWrapper


def main():
    envname = 'LunarLander-v2'
    env = gym.make(envname)
    env = HistoryWrapper(env, history_length=500)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_steps = env.spec.max_episode_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO(CustomActorCriticPolicy, env, verbose=1)

    model.learn(100, progress_bar=True)

    env.close()

    vec_env = gym.make(envname, render_mode='human')
    vec_env = HistoryWrapper(vec_env, history_length=500)

    obs, info = vec_env.reset()
    for i in tqdm(range(1000)):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = vec_env.step(action)
        vec_env.render()
if __name__ == "__main__":
    main()

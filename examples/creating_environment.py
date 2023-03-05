import gymnasium as gym
import pandas as pd

import gym_trading

#%%
env = gym_trading.envs.TradingEnv(df=gym_trading.datasets.BITCOIN_USD_1H)
# env_fn = lambda: gym_trading.envs.TradingEnv(df=df=gym_trading.datasets.BITCOIN_USD_1H)
# env = gym.vector.SyncVectorEnv([env_fn for _ in range(16)])
#%%
obs, info = env.reset()
num_steps = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    num_steps += 1
    if terminated or truncated:
        break
env.render()

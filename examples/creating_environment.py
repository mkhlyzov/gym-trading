import pandas as pd

import gym
import gym_trading
#%%
df = pd.read_csv(
    'https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv',
    header=1
)
df.sort_values(by=['date'], inplace=True, ignore_index=True)
#%%
env = gym_trading.envs.TradingEnv(df=df)
# env_fn = lambda: gym_trading.envs.TradingEnv(df=df)
# env = gym.vector.SyncVectorEnv([env_fn for _ in range(16)])
#%%
obs = env.reset()
num_steps = 0
while True:
    action = env.action_space.sample()
    new_obs, reward, done, info = env.step(action)
    num_steps += 1
    if done:
        break
env.render()

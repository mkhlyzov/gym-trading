# gym-trading
Trading environment for Reinforcement Learning

In order to install run following commands in your home directory:

``` bash
git clone https://github.com/mkhlyzov/gym-trading  
pip install -e ./gym-trading
```

Currently only editable install is supported due to lack of support for Data Files on developer's end.

Basic example can be found in [examples](examples/) folder.

Single observation type is a dictionary. One can use gymnasium.wrappers.FlattenObservation wrapper to flatten the observations.

---
```Python
import gym_trading

df = gym_trading.datasets.BITCOIN_USD_1H

env = gym_trading.envs.TradingEnv(
    df=df,
    max_episode_steps=400,  # Can also use time based, e.g. '24H'
    window_size=10,
    comission_fee=0.0007,   # Trading comission, 0.01 corresponds to 1% comission
    std_threshold=0.0010,   # Paramether to ocally scale time
    # scale=1,      # Sets observation step to a fixed number. Overrides std_threshold
    # reward_mode="trade",  # Default is per-step reward
)

obs, info = env.reset()

print(obs)
> ({'features': array([6.24808711e+04, 6.21325781e+04, 6.23635195e+04, 6.22393008e+04,
         6.17756484e+04, 6.12997305e+04, 6.06266289e+04, 6.21008008e+04,
         6.18421289e+04, 6.18561992e+04, 0.00000000e+00, 0.00000000e+00,
         1.00000000e+00]),
  'position': array([0.]),
  'price_change': array([0.]),
  'time_passed': array([0.]),
  'time_left': array([1.60943791])},
 {})

done = False
while not done:
    action = env.get_optimal_action()
    obs, r, term, trunc, info = env.step(action)
    done = (term or trunc)
env.render()
```
---

TradingEnv versions 2 and 3 require continuous data to work with. That is if the original pandas.DataFrame contains missing values due to, for example, lack of trades at that time, all missing values have to be filled. It is currently solved by using datetime as index and resampling the DataFrame during the Env creation (inside \_\_init__). The reason behind it is the "time distortion" trick. Candles in Env2 and Env3 are being locally resampled so that price volatility stays approximately constant. TradingEnv version 1 works with candles as they are, so data continuity and usage of datetime as index are not required.

---
TradingEnv2 determines current price volatility at the beginning of each episode and assumes volatility stays the same during the entire episode (which is not true of course but it is an approximation). Then DataFrame part that is related to current episode is being resampled as whole to ensure volatility hits predetermined level. For Example if the original Dataframe has 1-minute resolution, at the start of episode one it can be resampled to 7-minute resolutions for relatively fast-changing assets and at the start of episode two to 31-minute resolution for slow-changing assets.

---
TradingEnv3 carries idea similar to TradingEnv2 but adresses the incorrect assumption of price volatility staying the same for the duration of the whole episode. The resamplings happen at each step independently of each other. For example, in order to preserve constant volatility at moment {t} I want to treat data as 11-minute candles. Then to observe price history for the last 10 prices I look 11 minutes into the past, 22 minutes into the past, 33, 44 and so on. When decision about action (short flat long) is made, the next observed moment {t+1} will be 11 minutes into the future. But at {t+1} the volatility might be different and in order to compensate for it I now want to treat data as 12-minute candes instead of 11-minute candles. Or as 10-minute depending on actual data. Then if at {t+1} the decision is to use 12-minute candles, price history will look like this: 12, 24, 36,... and the next observation {t+2} will correspond to a 12 minute jump.

In this example {t+1} in 11 minutes ahead of {t} and {t+2} is 12 minutes ahead of {t+1}.
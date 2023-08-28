# gym-trading
Trading environment for Reinforcement Learning

In order to install run following commands in your home directory:

>git clone https://github.com/mkhlyzov/gym-trading  
>pip install -e ./gym-trading

Currently only editable install is supported due to lack of support for Data Files on developer's end. (Going to be fixed, but not the first priority)

Basic example can be found in [examples](examples/) folder.

Single observation type is a dictionary. One can use gymnasium.wrappers.FlattenObservation wrapper to flatten the observations.

---

TradingEnv versions 2 and 3 require continuous data to work with. That is if the original pandas.DataFrame contains missing values due to, for example, lack of trades at that time, all missing values have to be filled. It is currently solved by using datetime as index and resampling the DataFrame during the Env creation (inside \_\_init__). The reason behind it is the "time distortion" trick. Candles in Env2 and Env3 are being locally resampled so that price volatility stays approximately constant. TradingEnv version 1 works with candles as they are, so data continuity and usage of datetime as index are not required.

---
TradingEnv2 determines current price volatility at the beginning of each episode and assumes volatility stays the same during the entire episode (which is not true of course but it is an approximation). Then DataFrame part that is related to current episode is being resampled as whole to ensure volatility hits predetermined level. For Example if the original Dataframe has 1-minute resolution, at the start of episode one it can be resampled to 7-minute resolutions for relatively fast-changing assets and at the start of episode two to 31-minute resolution for slow-changing assets.

---
TradingEnv3 carries idea similar to TradingEnv2 but adresses the incorrect assumption of price volatility staying the same for the duration of the whole episode. The resamplings happen at each step independently of each other. For example, in order to preserve constant volatility at moment {t} I want to treat data as 11-minute candles. Then to observe price history for the last 10 prices I look 11 minutes into the past, 22 minutes into the past, 33, 44 and so on. When decision about action (short flat long) is made, the next observed moment {t+1} will be 11 minutes into the future. But at {t+1} the volatility might be different and in order to compensate for it I now want to treat data as 12-minute candes instead of 11-minute candles. Or as 10-minute depending on actual data. Then if at {t+1} the decision is to use 12-minute candles, price history will look like this: 12, 24, 36,... and the next observation {t+2} will correspond to a 12 minute jump.

In this example {t+1} in 11 minutes ahead of {t} and {t+2} is 12 minutes ahead of {t+1}.
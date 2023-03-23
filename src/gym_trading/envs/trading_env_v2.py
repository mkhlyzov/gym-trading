from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import gymnasium
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .trading_env import Position, TradingEnv


class TradingEnv2(TradingEnv):
    """
    It's a gymnasium environment that takes a dataframe of stock prices
    and allows you to trade on it.
    Dynamically resamples timeframe on start of a new episode to ensure
    semi-consatnt volatility of price. Works best when original dataframe
    has a 1-minute resolution.

    df - expected to be indexed by pd.DatetimeIndex
    """

    _std_window: str     # Window for calculating volatility on episode start
    _std_threshold: float

    def __init__(
        self,
        df: pd.DataFrame,
        max_episode_steps: Union[int, str] = "14D",
        process_data: Callable = None,
        window_size: int = 20,
        comission_fee: float = 0.001,
        std_threshold: float = 0.0020,
        std_window: str = "7D",
    ) -> None:
        self.df = df.resample((df.index[1:] - df.index[:-1]).min()).last()
        self.max_episode_steps = max_episode_steps
        self._std_threshold = std_threshold
        self._std_window = std_window
        self._comission_fee = comission_fee
        self._process_data = process_data if process_data is not None \
            else lambda x: self.default_preprocessor(x, window_size, std_threshold)
        
        self.reset()    # In order to call get_observation() for spaces
        # spaces
        self.action_space = gymnasium.spaces.Discrete(len(Position))
        self.observation_space = gymnasium.spaces.Dict(
            {
                "features": gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=self._get_observation()["features"].shape,
                    dtype=float,
                ),
                "price_change": gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=float
                ),
                "position": gymnasium.spaces.Box(0, 1, shape=(1,), dtype=float),
                "time_left": gymnasium.spaces.Box(0, 1, shape=(1,), dtype=float),
            }
        )

    @staticmethod
    def default_preprocessor(df: pd.DataFrame, window: int, std: float):
        mask = range(window)
        price = df.close.fillna(method="ffill")

        dp = (price.shift(1) - price) / (price.shift(1) + price)
        features = pd.concat([dp.shift(i) for i in mask], axis=1).fillna(0).values

        # features = pd.concat(
        #     [df.close.shift(i) / df.close for i in mask], axis=1
        # ).fillna(0).values

        return price, features / std
    
    def reset(self, idx_start=None, **kwargs) -> Tuple[Any, Dict]:
        super(TradingEnv, self).reset(**kwargs)

        idx1 = self.df.index[0] + pd.Timedelta(self._std_window)
        idx2 = self.df.index[-1] - pd.Timedelta("14D")  #=========================HARDCODED=VALUE===========

        start_idx = pd.Timestamp(idx_start) if idx_start is not None else \
            self.df[idx1:idx2].index[np.random.randint(len(self.df[idx1:idx2].index))]

        p = self.df.close[start_idx - pd.Timedelta(self._std_window):start_idx]
        dp = (p.shift(1) - p)/  (p.shift(1) + p)
        if dp.isna().mean() > 0.5:
            if idx_start is not None:
                raise RuntimeError("Too many NaN values, can't determine optimal scale")
            return self.reset()
        std = dp.std()
        scale = int(max(np.rint((self._std_threshold / std)**2), 1)) # Number of candles to  combine
        step = self.df.index[-1] - self.df.index[-2]

        episode_duration = pd.Timedelta(self.max_episode_steps) if isinstance(self.max_episode_steps, str) \
            else self.max_episode_steps * scale * step
        end_idx = start_idx + episode_duration
        end_idx = min(end_idx, self.df.index[-1]) # in case idx2 estimation was bad

        resampling_func = dict(
            open = "first",
            close = "last",
            high = "max",
            low = "min",
            volume = "sum",
        )
        resampling_func = {c: resampling_func[c] for c in self.df.columns}

        df = self.df[
            start_idx - pd.Timedelta(self._std_window):end_idx + step * scale * 2
        ].resample(step * scale, offset=0).aggregate(resampling_func)
        offset = (df[start_idx:].index - start_idx)[0]
        df = self.df[
            start_idx - pd.Timedelta(self._std_window):end_idx + step * scale * 2
        ].resample(step * scale, offset=-offset).aggregate(resampling_func)

        if df.close[start_idx:end_idx].isna().mean() > 0.3:
            # unlucky guess with too many missing values for the episode
            return self.reset()

        self.prices, self.signal_features = self._process_data(df)

        self._start_tick = (self.prices.index < start_idx).sum()
        self._end_tick = (self.prices.index < end_idx).sum()
        self._current_tick = self._start_tick

        self._last_trade_tick = None
        self._position = Position.FLAT
        self._old_position = None
        self._position_history = []
        self._total_reward = 0.
        self._total_profit = 1.

        return self._get_observation(), {}


if __name__ == "__main__":
    fname = "~/Downloads/archive/btcusd.csv"
    df = pd.read_csv(fname)
    df.time = pd.to_datetime(df.time, unit="ms")
    df.set_index("time", inplace=True)

    env = TradingEnv2(df["2014":], max_episode_steps=500, window_size=20)
    # obs, _ = env.reset('2018-09-10 07:32:00')
    obs, _ = env.reset()
    done = False
    while not done:
        # action = env.action_space.sample()
        action = env.get_optimal_action()
        obs, _, done, _, _ = env.step(action)
    print(env._total_profit, env._total_reward)
    # env.render()

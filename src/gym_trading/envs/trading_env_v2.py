from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd

from .base_env import BaseTradingEnv, Position


class TradingEnv2(BaseTradingEnv):
    """
    It's a gymnasium environment that takes a dataframe of stock prices
    and allows you to trade on it.
    Dynamically resamples timeframe on start of a new episode to ensure
    semi-consatnt volatility of price. Works best when original dataframe
    has a 1-minute resolution.

    df - expected to be indexed by pd.DatetimeIndex
    """

    _std_window: str  # Window for calculating volatility on episode start
    _std_threshold: float

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        max_episode_steps: Union[int, str] = "14D",
        process_data: Callable = None,
        window_size: int = 20,
        comission_fee: float = 0.001,
        std_threshold: float = 0.0020,
        std_window: str = "7D",
        reward_mode: str = "step",
    ) -> None:
        self._std_threshold = std_threshold
        self._std_window = std_window
        self._process_data = (
            process_data
            if process_data is not None
            else lambda x: self.default_preprocessor(x, window_size, std_threshold)
        )
        super().__init__(
            df=df,
            max_episode_steps=max_episode_steps,
            window_size=window_size,
            comission_fee=comission_fee,
            reward_mode=reward_mode,
        )

    @staticmethod
    def default_preprocessor(
        df: pd.DataFrame, window: int, std: float
    ) -> Tuple[pd.Series, np.ndarray]:
        mask = range(window)
        price = df.close.ffill()    # equivalent of df.close.fillna(method="ffill")

        # dp = (price - price.shift(1)) / (price + price.shift(1))
        dp = np.log(price / price.shift(1))
        features = pd.concat([dp.shift(i) for i in mask], axis=1).fillna(0).values

        # features = pd.concat(
        #     [df.close.shift(i) / df.close for i in mask], axis=1
        # ).fillna(0).values

        return price, features / std

    def _get_features(self) -> np.ndarray:
        return self.signal_features[self._current_tick]

    def reset(self, idx_start=None, **kwargs) -> Tuple[Any, Dict]:
        super(BaseTradingEnv, self).reset(**kwargs)

        idx1 = self.df.index[0] + pd.Timedelta(self._std_window)
        idx2 = self.df.index[-1] - pd.Timedelta(
            "14D"
        )  # =========================HARDCODED=VALUE===========

        start_idx = (
            pd.Timestamp(idx_start)
            if idx_start is not None
            else self.df[idx1:idx2].index[
                np.random.randint(len(self.df[idx1:idx2].index))
            ]
        )

        p = self.df.close[start_idx - pd.Timedelta(self._std_window) : start_idx]
        dp = (p.shift(1) - p) / (p.shift(1) + p)
        if dp.isna().mean() > 0.5:
            if idx_start is not None:
                raise RuntimeError("Too many NaN values, can't determine optimal scale")
            return self.reset()
        std = dp.std()
        scale = int(
            max(np.rint((self._std_threshold / std) ** 2), 1)
        )  # Number of candles to  combine
        step = self.df.index[-1] - self.df.index[-2]

        episode_duration = (
            pd.Timedelta(self.max_episode_steps)
            if isinstance(self.max_episode_steps, str)
            else self.max_episode_steps * scale * step
        )
        end_idx = start_idx + episode_duration
        end_idx = min(end_idx, self.df.index[-1])  # in case idx2 estimation was bad

        resampling_func = dict(
            open="first",
            close="last",
            high="max",
            low="min",
            volume="sum",
        )
        resampling_func = {
            c: resampling_func.get(c)
            if c in resampling_func
            else ("sum" if "volume" in c.lower() else "first")
            for c in self.df.columns
        }

        df = (
            self.df[
                start_idx - pd.Timedelta(self._std_window) : end_idx + step * scale * 2
            ]
            .resample(step * scale, offset=0)
            .aggregate(resampling_func)
        )
        offset = (df[start_idx:].index - start_idx)[0]
        df = (
            self.df[
                start_idx - pd.Timedelta(self._std_window) : end_idx + step * scale * 2
            ]
            .resample(step * scale, offset=-offset)
            .aggregate(resampling_func)
        )

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
        self._total_reward = 0.0
        self._total_profit = 1.0

        self._map_optimal_actions()

        return self._get_observation(), {}


if __name__ == "__main__":
    fname = "~/Downloads/archive/btcusd.csv"
    df = pd.read_csv(fname)
    df.time = pd.to_datetime(df.time, unit="ms")
    df.set_index("time", inplace=True)

    env = TradingEnv2(df=df["2014":], max_episode_steps=500, window_size=20)
    # obs, _ = env.reset('2018-09-10 07:32:00')
    obs, _ = env.reset()
    done = False
    while not done:
        # action = env.action_space.sample()
        action = env.get_optimal_action()
        obs, _, done, _, _ = env.step(action)
    print(env._total_profit, env._total_reward)
    # env.render()

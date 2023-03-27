from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import gymnasium
import numba
import numpy as np
import pandas as pd

from .trading_env import Position, TradingEnv


class TradingEnv3(TradingEnv):
    """
    It's a gymnasium environment that takes a dataframe of stock prices
    and allows you to trade on it.
    Third iteration of TradingEnv focuses on finding appropriate scale
    for each time moment and implements local discretization of
    price[i - n * step, ... , i - 2 * step, i - 1 * step, i]
    The next returned observation can have a different step, or the exact
    same.
    """

    prices: pd.Series

    # [_idx1, _idx2) represent a range of valid indexes to sample starting
    # index from on a start of a new episode
    _idx1: int
    _idx2: int

    _indices: np.ndarray

    def __init__(
        self,
        df: pd.DataFrame,
        max_episode_steps: Union[int, str] = "14D",
        window_size: int = 20,
        comission_fee: float = 0.0010,
        std_threshold: float = 0.0040,
        scale: int = None,
    ) -> None:
        self.df = self._set_scaling_step(df, std_threshold, window_size, scale)
        self.max_episode_steps = max_episode_steps
        self.window_size = window_size
        self._comission_fee = comission_fee

        self._idx1, self._idx2 = self._get_idx1_idx2()
        self._std_threshold = std_threshold
        self._scale = scale

        self.reset()  # In order to call get_observation() for spaces
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

    def _set_scaling_step(
        self, df: pd.DataFrame, std_threshold: float, window_size: int, scale: int
    ) -> pd.DataFrame:
        df = df.resample("1T").last()
        df = df.fillna(method="ffill")
        df["step"] = scale
        if scale is not None:
            return df

        def find_window_sizes(sequence, C) -> np.ndarray:
            cumsum = np.cumsum(sequence)
            indices = np.searchsorted(cumsum, cumsum - C)
            window_sizes = np.arange(len(sequence)) - indices
            window_sizes[window_sizes <= 0] = 0
            return window_sizes

        dp = (df["close"] - df["close"].shift(1)) / (df["close"] + df["close"].shift(1))
        steps = (
            find_window_sizes(dp**2, window_size * std_threshold**2) / window_size
        )
        df["step"] = np.rint(steps).astype(int)
        df["step"].replace(0, 1, inplace=True)
        return df
    
    def _get_idx1_idx2(self,) -> Tuple[int, int]:
        min_indices = np.arange(len(self.df)) - self.df["step"] * self.window_size
        idx1 = np.where(min_indices <= 0)[0][-1] + 1

        indices = (
            self._compute_indices_num_steps(
                np.flip(self.df["step"].values), 0, self.max_episode_steps * np.sqrt(2)
            )
            if isinstance(self.max_episode_steps, int)
            else self._compute_indices_max_sum(
                np.flip(self.df["step"].values),
                0,
                pd.Timedelta(self.max_episode_steps) / self.df.index.freq * np.sqrt(2),
            )
        )
        idx2 = len(self.df) - np.max(indices)

        return idx1, idx2


    @staticmethod
    @numba.jit(nopython=True)
    def _compute_indices_num_steps(
        steps: np.ndarray, start_idx: int, num_steps: int
    ) -> np.ndarray:
        indices = [start_idx]
        for _ in range(num_steps - 1):
            new_idx = indices[-1] + steps[indices[-1]]
            if new_idx >= len(steps):
                break
            indices.append(new_idx)
        return np.array(indices)

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_indices_max_sum(
        steps: np.ndarray, start_idx: int, max_sum: Union[int, float]
    ) -> np.ndarray:
        indices = [start_idx]
        sum_steps = 0
        while sum_steps < max_sum:
            step = steps[indices[-1]]
            new_idx = indices[-1] + step
            if new_idx >= len(steps):
                break
            indices.append(new_idx)
            sum_steps += step
        return np.array(indices)

    def _get_features(self) -> np.ndarray:
        current_idx = self._indices[self._current_tick]
        scale = self.df["step"][current_idx]
        indices = np.arange(-self.window_size, 1) * scale + current_idx
        close = self.df["close"].values[indices]
        dp = (close[1:] - close[:-1]) / (close[1:] + close[:-1])
        return dp / self._std_threshold

    def _get_observation(self) -> Tuple[np.array, float, bool, bool, dict]:
        price_change = 0
        if self._position != Position.FLAT:
            price_change = (
                self.prices[self._current_tick] - self.prices[self._last_trade_tick]
            ) / self.prices[self._last_trade_tick]
        position = self._position.value - 1
        features = self._get_features()
        time_left = np.clip((self._end_tick - self._current_tick) / 100.0, 0, 1)

        return {
            "features": features,
            "price_change": np.array([price_change], dtype=float),
            "position": np.array([position], dtype=float),
            "time_left": np.array([time_left], dtype=float),
        }

    def reset(self, idx_start: Union[int, str] = None, **kwargs) -> Tuple[Any, Dict]:
        super(TradingEnv, self).reset(**kwargs)

        start_idx = (
            idx_start
            if idx_start and isinstance(idx_start, int)
            else self.df.index.get_loc(pd.Timestamp(idx_start))
            if isinstance(idx_start, str)
            else np.random.randint(self._idx1, self._idx2)
        )
        p_hist = self.df["close"][
            start_idx - self.df["step"][start_idx] * self.window_size : start_idx
        ]
        if (p_hist == p_hist.shift(1)).mean() > 0.5:
            # too many missing values in recent history hence step is compromised
            return self.reset()

        self._indices = (
            self._compute_indices_num_steps(
                self.df["step"].values, start_idx, self.max_episode_steps + 1
            )
            if isinstance(self.max_episode_steps, int)
            else self._compute_indices_max_sum(
                self.df["step"].values,
                start_idx,
                pd.Timedelta(self.max_episode_steps) / self.df.index.freq,
            )
        )

        self.prices = self.df["close"][self._indices]
        if (self.prices.shift(1) == self.prices).mean() > 0.3:
            # unlucky guess with too many missing values for the episode
            return self.reset()

        self._start_tick = 0
        self._end_tick = len(self._indices) - 1
        self._current_tick = 0

        self._last_trade_tick = None
        self._position = Position.FLAT
        self._old_position = None
        self._position_history = []
        self._total_reward = 0.0
        self._total_profit = 1.0

        return self._get_observation(), {}


if __name__ == "__main__":
    df = pd.read_csv("~/Downloads/archive/btcusd.csv")
    df.time = pd.to_datetime(df.time, unit="ms")
    df.set_index("time", inplace=True)
    env = TradingEnv3(df, max_episode_steps=500, std_threshold=0.0020)
    env.reset()
    env.reset()
    done = False
    while not done:
        # action = env.action_space.sample()
        action = env.get_optimal_action()
        _, _, done, _, _ = env.step(action)
    print(env._total_reward, env._total_profit)

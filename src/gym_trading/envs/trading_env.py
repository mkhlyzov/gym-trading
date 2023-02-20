from enum import Enum
from typing import Any, Callable, Dict, Tuple

import gymnasium
import numpy as np
import pandas as pd


class Position(Enum):
    Short = 0
    Flat = 1
    Long = 2

    def opposite(self):
        return Position.Short if self == Position.Long else Position.Long


class TradingEnv(gymnasium.Env):
    metadata = {}

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        episode_length: int = 24 * 14,
        window_size: int = 20,
        comission_fee: float = 7e-4,
        preprocessor: Callable = None,
    ) -> None:
        self.df = df
        self.episode_length = episode_length
        self.window_size = window_size
        self._comission_fee = comission_fee
        self.max_episode_steps = episode_length
        self.prices, self.signal_features = self._process_data()

        self.reset()

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

    def _process_data(self):
        prices = self.df.loc[:, "close"].to_numpy()

        diff = np.insert(np.diff(prices), 0, 0)
        relative_diff = diff / prices

        def shift(xs, n):
            if n == 0:
                return xs
            assert n > 0
            e = np.empty_like(xs)
            e[:n] = np.nan
            e[n:] = xs[:-n]
            return e

        signal_features = np.stack(
            [shift(relative_diff, i) for i in range(self.window_size)], axis=1
        )

        return prices, signal_features

    def _get_observation(self) -> Dict[str, Any]:
        price_change = 0
        if self._position != Position.Flat:
            price_change = (
                self.prices[self._current_tick] - self.prices[self._last_trade_tick]
            ) / self.prices[self._last_trade_tick]
        position = self._position.value / 2.0
        features = self.signal_features[self._current_tick]
        time_left = np.clip((self._end_tick - self._current_tick) / 100.0, 0, 1)

        return {
            "features": features,
            "price_change": np.array([price_change]),
            "position": np.array([position]),
            "time_left": np.array([time_left]),
        }

    def _calculate_reward(self, action) -> float:
        new_price = self.prices[self._current_tick]
        old_price = self.prices[self._current_tick - 1]
        new_pos = action.value
        old_pos = self._position.value

        reward = np.log(new_price / old_price) * (new_pos - 1)
        reward -= self._comission_fee * abs(new_pos - old_pos)
        return reward

    def _update_profit(self, action) -> None:
        trade_close = False
        if self._position != Position.Flat and action != self._position:
            trade_close = True

        if trade_close:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Position.Long:
                shares = (
                    self._total_profit * (1 - self._comission_fee)
                ) / last_trade_price
                self._total_profit = (shares * (1 - self._comission_fee)) * current_price
            elif self._position == Position.Short:
                shares = (self._total_profit * (1 - self._comission_fee)) / current_price
                self._total_profit = (
                    shares * (1 - self._comission_fee)
                ) * last_trade_price

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        # obs, reward, termination, truncation, info ?
        action = Position(action)
        self._done = (self._current_tick + 1) >= self._end_tick
        if self._done:
            action = Position.Flat
        self._position_history.append(action)

        self._update_profit(action)
        if self._position != action:
            self._last_trade_tick = self._current_tick

        self._current_tick += 1
        reward = self._calculate_reward(action)
        self._total_reward += reward

        self._position = action

        return (self._get_observation(), reward, self._done, False, {})

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict]:
        super().reset(**kwargs)

        self._start_tick = self.np_random.integers(
            self.window_size, len(self.prices) - self.episode_length
        )
        self._end_tick = self._start_tick + self.episode_length
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Position.Flat
        self._position_history = []
        self._total_reward = 0
        self._total_profit = 1
        self.history = []

        return self._get_observation(), {}

    def close(self) -> None:
        pass

    def render(self) -> None:
        from matplotlib import pyplot as plt

        plt.style.use("seaborn")
        plt.figure(figsize=(25, 10), dpi=200)

        ticks = np.arange(self._start_tick, self._end_tick)
        prices = self.prices[ticks]
        ticks -= ticks.min()

        plt.plot(ticks, prices, "b", alpha=0.3)
        plt.plot(ticks, prices, "b.", alpha=0.3)

        for tick, position in zip(ticks, self._position_history):
            if position == Position.Short:
                plt.plot([tick], [prices[tick]], "ro", alpha=0.9)
            elif position == Position.Flat:
                plt.plot([tick], [prices[tick]], "bo", alpha=0.3)
            elif position == Position.Long:
                plt.plot([tick], [prices[tick]], "go", alpha=0.9)

        info = "total profit: {:.3f}; idx_start: {};".format(
            self._total_profit, self._start_tick
        )
        plt.title(info, fontsize=20)
        plt.xlabel("candle #")
        plt.ylabel("stock close price")

        plt.show()

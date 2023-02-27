from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import gymnasium
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Position(Enum):
    """
    The Position class is an enumeration of the possible positions
    that a trader can take in the market
    """

    SHORT = 0
    FLAT = 1
    LONG = 2


class TradingEnv(gymnasium.Env):
    """
    It's a gymnasium environment that takes a dataframe of stock prices
    and allows you to trade on it
    """

    metadata = {}

    action_space: gymnasium.spaces.Discrete
    observation_space: gymnasium.spaces.Dict
    reward_range: Tuple[float, float]

    max_episode_steps: int

    prices: pd.Series
    features: np.ndarray

    _total_profit: float
    _position: float
    _old_position: float
    _position_history: List[Position]
    _last_trade_tick: int

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        max_episode_steps: int = 24 * 14,
        window_size: int = 20,
        comission_fee: float = 7e-4,
        process_data: Callable = None,
    ) -> None:
        self.df = df
        if "date" in self.df.columns:
            self.df = self.df.set_index("date")

        self.max_episode_steps = max_episode_steps
        self.window_size = window_size
        self._comission_fee = comission_fee
        self.max_episode_steps = max_episode_steps

        self.prices, self.signal_features = (
            process_data(self) if process_data else self._process_data()
        )
        if self.prices.shape[0] != self.signal_features.shape[0]:
            raise ValueError("signal_features and prices have different shapes")

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

    def _process_data(self) -> Tuple[pd.Series, np.ndarray]:
        prices = self.df.close

        mask = list(range(self.window_size))
        signal_features = (
            pd.concat(
                [
                    (self.df.close.shift(mask[i]) - self.df.close.shift(mask[i + 1]))
                    / self.df.close.shift(mask[i])
                    for i, _ in enumerate(mask[:-1])
                ],
                axis=1,
            )
            .fillna(0)
            .to_numpy()
        )

        return prices, signal_features

    def _get_observation(self) -> Dict[str, Any]:
        price_change = 0
        if self._position != Position.FLAT:
            price_change = (
                self.prices[self._current_tick] - self.prices[self._last_trade_tick]
            ) / self.prices[self._last_trade_tick]
        position = self._position.value - 1
        features = self.signal_features[self._current_tick]
        time_left = np.clip((self._end_tick - self._current_tick) / 100.0, 0, 1)

        return {
            "features": features,
            "price_change": np.array([price_change], dtype=float),
            "position": np.array([position], dtype=float),
            "time_left": np.array([time_left], dtype=float),
        }

    def _calculate_reward(self) -> float:
        new_price = self.prices[self._current_tick]
        old_price = self.prices[self._current_tick - 1]
        new_pos = self._position.value
        old_pos = self._old_position.value

        reward = np.log(new_price / old_price) * (new_pos - 1)
        reward -= self._comission_fee * abs(new_pos - old_pos)
        return reward

    def _update_profit_on_deal_close(self) -> None:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        if self._old_position == Position.LONG:
            # Closing LONG position
            last_trade_price *= 1 + self._comission_fee
            current_price *= 1 - self._comission_fee
            shares = self._total_profit / last_trade_price
            self._total_profit = shares * current_price
        elif self._old_position == Position.SHORT:
            # Closing SHORT position
            last_trade_price *= 1 - self._comission_fee
            current_price *= 1 + self._comission_fee
            shares = self._total_profit / current_price
            self._total_profit = shares * last_trade_price

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        done = (self._current_tick + 1) >= self._end_tick
        next_position = Position(action) if not done else Position.FLAT
        self._position_history.append(next_position)

        if self._position != next_position:
            if self._position != Position.FLAT:
                self._update_profit_on_deal_close()
            self._last_trade_tick = self._current_tick

        self._current_tick += 1
        self._old_position = self._position
        self._position = next_position
        reward = self._calculate_reward()
        self._total_reward += reward

        return (self._get_observation(), reward, done, False, {})

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict]:
        super().reset(*args, **kwargs)

        self._start_tick = self.np_random.integers(
            self.window_size, len(self.prices) - self.max_episode_steps
        )
        self._end_tick = self._start_tick + self.max_episode_steps
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Position.FLAT
        self._old_position = None
        self._position_history = []
        self._total_reward = 0
        self._total_profit = 1

        return self._get_observation(), {}

    def close(self) -> None:
        pass

    def _get_optimal_action(self) -> Position:
        s = np.sign(
            self.prices[self._current_tick + 1] - self.prices[self._current_tick]
        )
        if s == 0:
            return self._position

        threshold = (1 + self._comission_fee) / (1 - self._comission_fee)
        p_ = self.prices[self._current_tick]
        j = self._current_tick + 1
        while j <= self._end_tick:
            p_ = s * max(s * p_, s * self.prices[j])
            delta_p = (p_ / self.prices[self._current_tick]) ** s
            drawback = (p_ / self.prices[j]) ** s
            if drawback > threshold or drawback > delta_p:
                break
            j += 1

        if delta_p < threshold:
            return self._position
        return Position(1 + s)

    def render(self) -> None:
        plt.style.use("seaborn")
        plt.figure(figsize=(25, 10), dpi=200)

        ticks = np.arange(self._start_tick, self._end_tick)
        prices = self.prices[ticks]
        ticks -= ticks.min()

        plt.plot(ticks, prices, "b", alpha=0.3)
        plt.plot(ticks, prices, "b.", alpha=0.3)

        for tick, position in zip(ticks, self._position_history):
            if position == Position.SHORT:
                plt.plot([tick], [prices[tick]], "ro", alpha=0.9)
            elif position == Position.FLAT:
                plt.plot([tick], [prices[tick]], "bo", alpha=0.3)
            elif position == Position.LONG:
                plt.plot([tick], [prices[tick]], "go", alpha=0.9)

        info = f"total profit: {self._total_profit:.3f}; idx_start: {self._start_tick};"
        plt.title(info, fontsize=20)
        plt.xlabel("candle #")
        plt.ylabel("stock close price")

        plt.show()

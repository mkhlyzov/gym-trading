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


class BaseTradingEnv(gymnasium.Env):
    """
    Base class that represents TradingEnv functionality and implements
    standart logic: stepping, reward / profit calculation, rendering.
    Derived environments only have to implement reset and _get_features
    methods.
    self.reset has to build self.prices and set episod boundaries on
    self.prices through _start_tick and _end_tick.
    """

    metadata = {}

    action_space: gymnasium.spaces.Discrete
    observation_space: gymnasium.spaces.Dict
    reward_range: Tuple[float, float]

    max_episode_steps: int

    prices: pd.Series
    features: np.ndarray

    _start_tick: int
    _end_tick: int
    _current_tick: int

    _total_profit: float
    _total_reward: float
    _position: Position
    _old_position: Position
    _position_history: List[Position]
    _last_trade_tick: int
    _old_profit: float

    _calculate_raward: Callable

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        max_episode_steps: int = 24 * 14,
        window_size: int = 20,
        comission_fee: float = 7e-4,
        reward_mode: str = "step",
        **kwargs,
    ) -> None:
        self.df = df
        self.max_episode_steps = max_episode_steps
        self.window_size = window_size
        self._comission_fee = comission_fee

        if reward_mode == "step":
            self._calculate_reward = self._calculate_reward_per_step
        elif reward_mode == "trade":
            self._calculate_reward = self._calculate_reward_per_trade
        else:
            raise ValueError(f"Unsupported reward mode: {reward_mode}")

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
                    -np.pi / 2, np.pi / 2, shape=(1,), dtype=float
                ),
                "position": gymnasium.spaces.Box(-1, 1, shape=(1,), dtype=float),
                "time_left": gymnasium.spaces.Box(0, np.inf, shape=(1,), dtype=float),
            }
        )

    def _get_features(self) -> np.ndarray:
        raise NotImplementedError("Derived class has to implement this method")
    
    def _get_observation(self) -> Dict[str, Any]:
        price_change = 0
        if self._position != Position.FLAT:
            # applying np.log to make symmetrical, np.arctan to make it bounded
            price_change = np.log(
                self.prices[self._current_tick] / self.prices[self._last_trade_tick]
            )
            price_change = np.arctan(price_change * 100)
        position = self._position.value - 1
        features = self._get_features()
        time_left = np.log(1 + (self._end_tick - self._current_tick) / 100)

        return {
            "features": features,
            "price_change": np.array([price_change], dtype=float),
            "position": np.array([position], dtype=float),
            "time_left": np.array([time_left], dtype=float),
        }

    def _calculate_reward_per_step(self) -> float:
        new_price = self.prices[self._current_tick]
        old_price = self.prices[self._current_tick - 1]
        new_pos = self._position.value
        old_pos = self._old_position.value

        reward = np.log(new_price / old_price) * (new_pos - 1)
        reward -= self._comission_fee * abs(new_pos - old_pos)
        return reward
    
    def _calculate_reward_per_trade(self) -> float:
        return np.log(self._total_profit / self._old_profit)

    def _update_profit_on_deal_close(self) -> None:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        if self._position == Position.LONG:
            # Closing LONG position
            last_trade_price *= 1 + self._comission_fee
            current_price *= 1 - self._comission_fee
            shares = self._total_profit / last_trade_price
            self._total_profit = shares * current_price
        elif self._position == Position.SHORT:
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

        self._old_profit = self._total_profit
        if self._position != next_position:
            if self._position != Position.FLAT:
                self._update_profit_on_deal_close()
            self._last_trade_tick = self._current_tick

        if not done:
            self._current_tick += 1
        self._old_position = self._position
        self._position = next_position
        reward = self._calculate_reward()
        self._total_reward += reward

        return (self._get_observation(), reward, done, False, {})

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        raise NotImplementedError("Derived class has to implement this method")

    def close(self) -> None:
        pass

    def get_optimal_action(self) -> Position:
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
    
    def get_max_profit(self) -> float:
        threshold = (1 + self._comission_fee) / (1 - self._comission_fee)
        profit = 1.
        i = self._start_tick

        while i < self._end_tick:
            s = np.sign(self.prices[i + 1] - self.prices[i])
            if s == 0:
                i += 1
                continue
            p_ = self.prices[i]
            idx_extremum = i
            j = i + 1
            while j <= self._end_tick:
                p_ = s * max(s * p_, s * self.prices[j])
                if np.isclose(self.prices[j], p_):
                    idx_extremum = j
                delta_p = (p_ / self.prices[i]) ** s
                drawback = (p_ / self.prices[j]) ** s
                if drawback > threshold or drawback > delta_p:
                    break
                j += 1

            if delta_p >= threshold:
                profit *= (delta_p / threshold)

            i = idx_extremum

        return profit

    def render(self) -> None:
        plt.style.use("seaborn")
        plt.figure(figsize=(25, 10), dpi=200)

        index = self.prices.index[self._start_tick: self._end_tick]
        df = pd.DataFrame(dict(
            price=self.prices[index[:len(self._position_history)]],
            position=self._position_history,
        ))

        plt.plot(self.prices[index], "b", alpha=0.3)
        plt.plot(self.prices[index], "b.", alpha=0.3)
        plt.plot(df.price[df.position == Position.SHORT], "ro", alpha=0.9)
        plt.plot(df.price[df.position == Position.LONG], "go", alpha=0.9)

        info = f"total profit: {self._total_profit:.3f};    " +\
            f"idx_start: {self.prices.index[self._start_tick]};   " +\
            f"max possible profit: {self.get_max_profit():.3f};   "
        if self.prices.index.freq is not None:
            info += f"scale: {self.prices.index.freq};"
        if isinstance(index, pd.DatetimeIndex):
            info += f"\nepisode duration: {index[-1] - index[0]}"
        plt.title(info, fontsize=20)
        plt.xlabel("datetime")
        plt.ylabel("Price")

        plt.show()
        
import datetime
from numbers import Integral
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import gymnasium
import numba
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

    self.reset has to build self.price and set episod boundaries on
    self.price through _idx_first and _idx_last.
    """

    metadata = {}

    action_space: gymnasium.spaces.Discrete
    observation_space: gymnasium.spaces.Dict
    reward_range: Tuple[float, float]

    max_episode_steps: int
    comission_fee: float

    df: pd.DataFrame    # Holdas all historical data available
    price: pd.Series    # Price observations for one episode
    _indices: np.ndarray    # price = df[_indices]

    _idx_first: int     # Represents episode boundary
    _idx_last: int      # Represents episode boundary
    _idx_now: int       # _idx_now = _idx_first + [0, 1, 2, 3, ...]
    # idx = _indices[_idx_now]

    # Represent a range of valid indexes to sample starting
    # index from on a start of a new episode
    _idx1: int  
    _idx2: int

    # Skip episode if it contains too many (%) NaN values (df["close"])
    # Safety mechanism in case _idx12 were defined poorly
    NAN_PCT_TOLERANCE: float = 0.25

    _position: Position
    _old_position: Position
    _position_history: List[Position]
    _total_profit: float
    _total_reward: float
    _idx_last_trade: int
    _old_profit: float
    _optimal_positions: np.ndarray

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
        self._setup_dataframe(df)
        self._setup_reward(reward_mode)

        self.max_episode_steps = max_episode_steps
        self.window_size = window_size
        self.comission_fee = comission_fee

        self.reset()  # required for self._get_observation() call to work properly

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
                "position": gymnasium.spaces.Box(-1, 1, shape=(1,), dtype=float),
                "price_change": gymnasium.spaces.Box(
                    -np.pi / 2, np.pi / 2, shape=(1,), dtype=float
                ),
                "time_passed": gymnasium.spaces.Box(0, np.inf, shape=(1,), dtype=float),
                "time_left": gymnasium.spaces.Box(0, np.inf, shape=(1,), dtype=float),
            }
        )

    def _setup_dataframe(self, df: pd.DataFrame) -> None:
        """
        Method standartises DataFrame format.
        All column names get changed to lower cass and DF gets indexed by
        datetime64[ns] if possible.
        """
        df = df.rename(columns=lambda x: x.lower())

        for column in df.columns:
            if df[column].dtype == 'float64':
                df[column] = df[column].astype('float32')

            
        if not df.index.dtype == "datetime64[ns]":
            for column in df.columns:
                try:
                    # Try converting the column to datetime
                    converted = pd.to_datetime(df[column], errors="raise", unit="s")
                    if all(
                        (converted >= datetime.datetime(1971, 1, 1))
                        & (converted <= datetime.datetime.now())
                    ):
                        df[column] = converted
                        df.set_index(column, inplace=True)
                        break  # Stop if successful
                except ValueError:
                    pass
        if df.index.freq is None:
            df = df.resample(
                (df.index[1:] - df.index[:-1]).median()
            ).last()  # filling up missing rows

        df = df.ffill()     # equivalent of deprecated df.fillna(method="ffill")
        self.df = df

    def _get_features(self) -> np.ndarray:
        raise NotImplementedError("Derived class has to implement this method")

    def _get_observation(self) -> Dict[str, Any]:
        price_change = np.log(
            self.price.iloc[self._idx_now]
            / self.price.iloc[self._idx_last_trade]
        )
        price_change = np.arctan(price_change * 100)
        time_passed = np.log(1 + (self._idx_now - self._idx_last_trade) / 100)

        features = self._get_features()
        position = self._position.value - 1
        time_left = np.log(1 + (self._idx_last - self._idx_now) / 100)

        return {
            "features": features,
            "position": np.array([position], dtype=float),
            "price_change": np.array([price_change], dtype=float),
            "time_passed": np.array([time_passed], dtype=float),
            "time_left": np.array([time_left], dtype=float),
        }

    def _setup_reward(self, reward_mode: str) -> None:
        if reward_mode == "step":
            self._calculate_reward = self._calculate_reward_per_step
        elif reward_mode == "trade":
            self._calculate_reward = self._calculate_reward_per_trade
        elif reward_mode.startswith("mixed"):
            alpha = (
                0.1
                if len(reward_mode.split("_")) == 1
                else float(reward_mode.split("_")[-1])
            )
            self._calculate_reward = lambda: self._calculate_reward_mixed(alpha)
        elif reward_mode == "optimal_action":
            self._calculate_reward = self._calculate_raward_optimal_action
        else:
            raise ValueError(f"Unsupported reward mode: {reward_mode}")

    def _calculate_reward_per_step(self) -> float:
        new_price = self.price.iloc[self._idx_now]
        old_price = self.price.iloc[self._idx_now - 1]
        new_pos = self._position.value
        old_pos = self._old_position.value

        reward = np.log(new_price / old_price) * (new_pos - 1)
        reward -= self.comission_fee * abs(new_pos - old_pos)
        return reward

    def _calculate_reward_per_trade(self) -> float:
        return np.log(self._total_profit / self._old_profit)

    def _calculate_reward_mixed(self, alpha: float) -> float:
        r_step = self._calculate_reward_per_step()
        r_trade = self._calculate_reward_per_trade()
        return r_step * alpha + r_trade * (1 - alpha)

    def _calculate_raward_optimal_action(self) -> float:
        action_taken = self._position.value

        self._idx_now -= 1
        optimal_action = self.get_optimal_action().value
        self._idx_now += 1

        reward = 1.0 - abs(action_taken - optimal_action)
        return reward

    def _update_profit_on_deal_close(self) -> None:
        current_price = self.price.iloc[self._idx_now]
        last_trade_price = self.price.iloc[self._idx_last_trade]

        if self._position == Position.LONG:
            # Closing LONG position
            last_trade_price *= 1 + self.comission_fee
            current_price *= 1 - self.comission_fee
            shares = self._total_profit / last_trade_price
            self._total_profit = shares * current_price
        elif self._position == Position.SHORT:
            # Closing SHORT position
            last_trade_price *= 1 - self.comission_fee
            current_price *= 1 + self.comission_fee
            shares = self._total_profit / current_price
            self._total_profit = shares * last_trade_price

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        done = (self._idx_now + 1) >= self._idx_last
        next_position = Position(action) if not done else Position.FLAT
        self._position_history.append(next_position)

        self._old_profit = self._total_profit
        if self._position != next_position:
            if self._position != Position.FLAT:
                self._update_profit_on_deal_close()
            self._idx_last_trade = self._idx_now

        if not done:
            self._idx_now += 1
        self._old_position = self._position
        self._position = next_position
        reward = self._calculate_reward()
        self._total_reward += reward

        return (self._get_observation(), reward, done, False, {})

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        raise NotImplementedError("Derived class has to implement this method")
    
    def reset(self, idx_start=None, **kwargs) -> Tuple[Any, Dict]:
        self._idx_now = self._idx_first
        self._idx_last_trade = self._idx_first
        self._position = Position.FLAT
        self._old_position = None
        self._position_history = []
        self._total_reward = 0.
        self._total_profit = 1.
        self._old_profit = None
        self._map_optimal_actions()

        return self._get_observation(), {}

    def close(self) -> None:
        pass

    @staticmethod
    @numba.njit
    def _find_next_best_price(
        prices: np.ndarray, min_profit: float
    ) -> Tuple[int, float, int]:
        """
        prices: numpy array of prices
        min_profit: minimal trade profit (before comission).
                    min_profit = 0.01 corresponds to 1% price change

        returns (next_trading_point_idx, delta_p, price_change_direction[-1, 0, 1])
        """
        if len(prices) < 2:
            return 1, 0., 0

        s = np.sign(prices[1] - prices[0])
        if s == 0:
            return 1, 0., 0

        p_ = prices[0]
        delta_p = 0.
        j = 1
        idx_extremum = j
        while j < len(prices):
            p_ = s * max(s * p_, s * prices[j])
            if np.isclose(prices[j], p_):
                idx_extremum = j
            delta_p = (p_ / prices[0]) ** s - 1
            drawback = (p_ / prices[j]) ** s - 1
            # if drawback > min_profit or drawback > delta_p:
            # if drawback > delta_p / 3:
            if drawback > delta_p or (delta_p > min_profit and drawback > min(delta_p * 0.33, min_profit)):
                break
            j += 1

        return idx_extremum, delta_p, s

    def _map_optimal_actions(self, comission_fee: float = None) -> None:
        """
        Maps the optimal actions for each time step based on the commission fee.

        Parameters:
            comission_fee (float, optional): The commission fee applied to each trade. Defaults to None.

        Returns:
            None
        """
        if comission_fee is None: comission_fee = self.comission_fee
        min_profit = (1 + comission_fee) / (1 - comission_fee) - 1

        # breaks if max_episode_steps is str, e.g. '14D'
        # _optimal_positions = np.ones(self.max_episode_steps, dtype=np.int8)

        if (
            not hasattr(self, '_optimal_positions')
            or len(self._optimal_positions) < len(self.price)
        ):
            self._optimal_positions = -np.ones(len(self.price), dtype=np.int8)

        i = i0 = self._idx_first
        while i < self._idx_last:
            idx_extremum, delta_p, s = self._find_next_best_price(
                self.price.to_numpy()[i : self._idx_last], min_profit
            )
            if delta_p <= min_profit:
                s = 0
            self._optimal_positions[i - i0 : i - i0 + idx_extremum] = 1 + s
            i += idx_extremum
    
    def _get_optimal_action_static(self) -> Position:
        optimal_state = self._optimal_positions[self._idx_now - self._idx_first]
        return Position(optimal_state)
    
    def _get_optimal_action_dynamic(self, comission_fee: float = None) -> Position:
        if comission_fee is None:
            comission_fee = self.comission_fee
        if self._idx_now + 1 >= self._idx_last:
            return Position(1)
        threshold = (1 + comission_fee) / (1 - comission_fee)
        _, delta_p, s = self._find_next_best_price(
            self.price.to_numpy()[self._idx_now : self._idx_last], threshold
        )
        
        if delta_p <= threshold:
            if (self._position.value - 1) * s > 0:
                return self._position
            return Position(1)
            return self._position
        return Position(1 + s)

    def get_optimal_action(self, comission_fee: float = None) -> Position:
        return self._get_optimal_action_static()

    def get_max_profit(self) -> float:
        min_profit = (1 + self.comission_fee) / (1 - self.comission_fee) - 1
        profit = 1.0
        i = self._idx_first

        while i < self._idx_last:
            j, delta_p, _ = self._find_next_best_price(
                self.price.to_numpy()[i : self._idx_last], min_profit
            )
            i += j
            if delta_p >= min_profit:
                profit *= (delta_p + 1) / (min_profit + 1)

        return profit

    def get_buy_and_hold(self) -> float:
        buy_and_hold = (
            self.price.iloc[self._idx_last - 1] / self.price.iloc[self._idx_first]
        )
        if buy_and_hold < 1:
            buy_and_hold = 1 / buy_and_hold
        buy_and_hold *= (1 - self.comission_fee) / (1 + self.comission_fee)
        return buy_and_hold

    def render(self) -> None:
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(25, 10), dpi=200)

        index = self.price.index[self._idx_first : self._idx_last]
        df = pd.DataFrame(
            dict(
                price=self.price[index[: len(self._position_history)]],
                position=self._position_history,
            )
        )

        plt.plot(self.price[index], "b", alpha=0.3)
        plt.plot(self.price[index], "b.", alpha=0.3)
        plt.plot(df.price[df.position == Position.SHORT], "ro", alpha=0.9)
        plt.plot(df.price[df.position == Position.LONG], "go", alpha=0.9)

        info = (
            f"total profit: {self._total_profit:.3f};    "
            + f"max possible profit: {self.get_max_profit():.3f};   "
            + f"B&H: {self.get_buy_and_hold():.3f};   "
            + f"fee: {self.comission_fee * 100}%;   "
        )
        if self.price.index.freq is not None:
            info += f"scale: {self.price.index.freq};"
        info += "\n"
        if isinstance(index, pd.DatetimeIndex):
            info += f"episode duration: {index[-1] - index[0]}   "
        info += f"idx_start: {self.price.index[self._idx_first]};"
        plt.title(info, fontsize=20)
        plt.xlabel("datetime")
        plt.ylabel("Price")

        plt.show()

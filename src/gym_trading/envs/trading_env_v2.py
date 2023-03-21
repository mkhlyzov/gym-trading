from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

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


class TradingEnv2(gymnasium.Env):
    """
    It's a gymnasium environment that takes a dataframe of stock prices
    and allows you to trade on it.
    Dynamically resamples timeframe on start of a new episode to ensure
    semi-consatnt volatility of price. Works best when original dataframe
    has a 1-minute resolution.

    df - expected to be indexed by pd.DatetimeIndex
    """
    metadata = {}

    action_space: gymnasium.spaces.Discrete
    observation_space: gymnasium.spaces.Dict
    reward_range: Tuple[float, float]

    max_episode_steps: Union[int, str]
    comission_fee: float

    df: pd.DataFrame
    prices: pd.Series
    signal_features: pd.DataFrame

    std_window: str     # Window for calculating volatility on episode start

    start_idx: pd.Timestamp
    end_idx: pd.Timestamp
    current_idx: pd.Timestamp
    idx_step: pd.Timedelta

    total_reward: float
    total_profit: float
    position: Position
    old_position: Position
    position_history: List[Position]
    last_trade_idx: pd.Timestamp


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
        self.std_threshold = std_threshold
        self.std_window = std_window
        self.comission_fee = comission_fee
        self.process_data = process_data if process_data is not None \
            else lambda x: self.default_preprocessor(x, window_size, std_threshold)
        
        self.reset()    # In order to call get_observation() for spaces
        # spaces
        self.action_space = gymnasium.spaces.Discrete(len(Position))
        self.observation_space = gymnasium.spaces.Dict(
            {
                "features": gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=self.get_observation()["features"].shape,
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
        features = pd.concat(
            [
                (price.shift(i + 1) - price.shift(i)) / (price.shift(i + 1) + price.shift(i))
                # df.close.shift(i) / df.close
                for i in mask
            ], axis=1
        ).fillna(0)
        return price, features / std
    
    def get_observation(self) -> Dict[str, Any]:
        price_change = 0
        if self.position != Position.FLAT:
            price_change = (
                self.prices[self.current_idx] - self.prices[self.last_trade_idx]
            ) / self.prices[self.last_trade_idx]
        position = self.position.value - 1
        features = self.signal_features.loc[self.current_idx].values
        time_left = np.clip((self.end_idx - self.current_idx) / self.idx_step / 100.0, 0, 1)

        return {
            "features": features,
            "price_change": np.array([price_change * 10], dtype=float), # multiply by 10 to normalize
            "position": np.array([position], dtype=float),
            "time_left": np.array([time_left], dtype=float),
        }
    
    def calculate_reward(self) -> float:
        new_pos = self.position.value
        old_pos = self.old_position.value
        reward = -self.comission_fee * abs(new_pos - old_pos)
        
        new_price = self.prices[self.current_idx]
        old_price = self.prices[self.current_idx - self.idx_step]
        reward += np.log(new_price / old_price) * (new_pos - 1)

        return reward
    
    def update_profit_on_deal_close(self) -> None:
        current_price = self.prices[self.current_idx]
        last_trade_price = self.prices[self.last_trade_idx]

        if self.position == Position.LONG:
            # Closing LONG position
            last_trade_price *= 1 + self.comission_fee
            current_price *= 1 - self.comission_fee
            shares = self.total_profit / last_trade_price
            self.total_profit = shares * current_price
        elif self.position == Position.SHORT:
            # Closing SHORT position
            last_trade_price *= 1 - self.comission_fee
            current_price *= 1 + self.comission_fee
            shares = self.total_profit / current_price
            self.total_profit = shares * last_trade_price

    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        # obs, reward, terminated, truncated, info
        done = (self.current_idx + self.idx_step) >= self.end_idx
        next_position = Position(action) if not done else Position.FLAT
        self.position_history.append(next_position)

        if self.position != next_position:
            if self.position != Position.FLAT:
                self.update_profit_on_deal_close()
            self.last_trade_idx = self.current_idx

        if not done:
            self.current_idx += self.idx_step
        self.old_position = self.position
        self.position = next_position
        reward = self.calculate_reward()
        self.total_reward += reward

        return (self.get_observation(), reward, done, False, {})

    def reset(self, start_idx=None, **kwargs) -> Tuple[Any, Dict]:
        super().reset(**kwargs)

        idx1 = self.df.index[0] + pd.Timedelta(self.std_window)
        idx2 = self.df.index[-1] - pd.Timedelta("14D")  #=========================HARDCODED=VALUE===========
        self.start_idx = self.df[idx1:idx2].sample().index[0] if start_idx is None else pd.Timestamp(start_idx)

        p = self.df.close[self.start_idx - pd.Timedelta(self.std_window):self.start_idx]
        dp = (p.shift(1) - p)/  (p.shift(1) + p)
        if dp.isna().mean() > 0.5:
            if start_idx is not None:
                raise RuntimeError("Too many NaN values, can't determine optimal scale")
            return self.reset()
        std = dp.std()
        scale = int(max(np.rint((self.std_threshold / std)**2), 1)) # Number of candles to  combine
        step = self.df.index[-1] - self.df.index[-2]

        episode_duration = pd.Timedelta(self.max_episode_steps) if isinstance(self.max_episode_steps, str) \
            else self.max_episode_steps * scale * step
        self.end_idx = self.start_idx + episode_duration

        self.end_idx = min(self.end_idx, self.df.index[-1]) # in case idx2 estimation was bad

        self.current_idx = self.start_idx
        self.idx_step = scale * step

        resampling_func = dict(
            open = "first",
            close = "last",
            high = "max",
            low = "min",
            volume = "sum",
        )
        for i in range(scale):
            df = self.df[
                self.start_idx - pd.Timedelta(self.std_window):self.end_idx
            ].resample(step * scale, offset=i*step).aggregate(resampling_func)
            if self.start_idx in df.index:
                break

        self.prices, self.signal_features = self.process_data(df)
        self.last_trade_idx = None
        self.position = Position.FLAT
        self.old_position = None
        self.position_history = []
        self.total_reward = 0.
        self.total_profit = 1.

        return self.get_observation(), {}
    
    def close(self) -> None:
        pass

    def get_optimal_action(self) -> Position:
        if self.current_idx + self.idx_step >= self.end_idx:
            return self.position
        s = np.sign(
            self.prices[self.current_idx + self.idx_step] - self.prices[self.current_idx]
        )
        if s == 0:
            return self.position

        threshold = (1 + self.comission_fee) / (1 - self.comission_fee)
        p_ = self.prices[self.current_idx]
        j = self.current_idx + self.idx_step
        while j <= self.end_idx:
            p_ = s * max(s * p_, s * self.prices[j])
            delta_p = (p_ / self.prices[self.current_idx]) ** s
            drawback = (p_ / self.prices[j]) ** s
            if drawback > threshold or drawback > delta_p:
                break
            j += self.idx_step

        if delta_p < threshold:
            return self.position
        return Position(1 + s)
    
    def get_max_profit(self) -> float:
        threshold = (1 + self.comission_fee) / (1 - self.comission_fee)
        profit = 1.
        i = self.start_idx

        while i + self.idx_step < self.end_idx:
            s = np.sign(self.prices[i + self.idx_step] - self.prices[i])
            if s == 0:
                i += self.idx_step
                continue
            p_ = self.prices[i]
            idx_extremum = i
            j = i + self.idx_step
            while j <= self.end_idx:
                p_ = s * max(s * p_, s * self.prices[j])
                if np.isclose(self.prices[j], p_):
                    idx_extremum = j
                delta_p = (p_ / self.prices[i]) ** s
                drawback = (p_ / self.prices[j]) ** s
                if drawback > threshold or drawback > delta_p:
                    break
                j += self.idx_step

            if delta_p >= threshold:
                profit *= (delta_p / threshold)

            i = idx_extremum

        return profit
    
    def render(self) -> None:
        plt.style.use("seaborn")
        plt.figure(figsize=(25, 10), dpi=200)

        index = self.prices[self.start_idx:self.end_idx].index
        if len(index) > len(self.position_history):
            index = index[:-1]
        df = pd.DataFrame(dict(
            price=self.prices[index],
            position=self.position_history,
        ))
    
        plt.plot(df.price, "b", alpha=0.3)
        plt.plot(df.price, "b.", alpha=0.3)
        plt.plot(df.price[df.position == Position.SHORT], "ro", alpha=0.9)
        # plt.plot(df.price[df.position == Position.FLAT], "bo", alpha=0.3)
        plt.plot(df.price[df.position == Position.LONG], "go", alpha=0.9)

        info = f"total profit: {self.total_profit:.3f};  idx_start: {self.start_idx};  \
            max possible profit: {self.get_max_profit():.3f};   scale={self.prices.index.freq};"
        plt.title(info, fontsize=20)
        plt.xlabel("datetime")
        plt.ylabel("Price")

        plt.show()


if __name__ == "__main__":
    fname = "~/Downloads/archive/btcusd.csv"
    df = pd.read_csv(fname)
    df.time = pd.to_datetime(df.time, unit="ms")
    df.set_index("time", inplace=True)

    def process_data(df):
        mask = range(20)
        features = pd.concat(
            [
                # (df.close.shift(i + 1) - df.close.shift(i)) / (df.close.shift(i + 1) + df.close.shift(i))
                df.close.shift(i) / df.close
                for i in mask
            ], axis=1
        )
        return df.close, features

    # env = TradingEnv2(df["2014":], 500, process_data=process_data)
    env = TradingEnv2(df["2014":], max_episode_steps="14D", window_size=20)
    obs, _ = env.reset('2018-08-18 17:16:00')
    done = False
    while not done:
        # action = env.action_space.sample()
        action = env.get_optimal_action()
        obs, _, done, _, _ = env.step(action)
    print(env.total_profit, env.total_reward)
    # env.render()

    # volat = []
    # for i in range(1000):
    #     env.reset()
    #     features = env.signal_features.loc[env.start_idx:env.end_idx].values
    #     std = features[features != 0].std()
    #     if not np.isnan(std):
    #         volat.append(std)
    # print(f"{np.mean(volat):.4f} ± {np.std(volat):.4f}")

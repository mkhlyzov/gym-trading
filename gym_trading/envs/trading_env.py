import sys
import os

import random
import datetime

import gym
from gym.utils import seeding
from enum import Enum
import numpy as np
import pandas as pd


class Positions(Enum):
    Short = 0
    Flat = 1
    Long = 2

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        comission_fee: float = 7e-4,
        window_size: int = 20,
        episode_length: int = 24 * 14
    ) -> None:
        self.df = df
        self.comission_fee = comission_fee
        self.window_size = window_size
        self.episode_length = episode_length
        self.prices, self.signal_features = self._process_data()

        self.reset()

        # spaces
        self.action_space = gym.spaces.Discrete(len(Positions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._get_observation().shape,
            dtype=np.float64
        )

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()

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
            [shift(relative_diff, i) for i in range(self.window_size)],
            axis=1
        )

        return prices, signal_features

    def _get_observation(self):
        price_change = 0
        if self._position != Positions.Flat:
            price_change = (self.prices[self._current_tick] - self.prices[self._last_trade_tick]) / self.prices[self._last_trade_tick]
        position = self._position.value / 3.
        features = self.signal_features[self._current_tick]
        return np.concatenate([
            features.flatten(),
            [price_change, position]
        ])

    def _calculate_reward(self, action):
        new_price = self.prices[self._current_tick]
        old_price = self.prices[self._current_tick - 1]
        new_pos = action.value
        old_pos = self._position.value

        reward = np.log(new_price / old_price) * (new_pos - 1)
        reward -= self.comission_fee * abs(new_pos - old_pos)
        return reward

    def _update_profit(self, action):
        trade_close = False
        if self._position != Positions.Flat and action != self._position:
            trade_close = True

        if trade_close:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.comission_fee)) / last_trade_price
                self._total_profit = (shares * (1 - self.comission_fee)) * current_price
            elif self._position == Positions.Short:
                shares = (self._total_profit * (1 - self.comission_fee)) / current_price
                self._total_profit = (shares * (1 - self.comission_fee)) * last_trade_price

    def step(self, action) -> (np.array, float, bool, dict):
        # obs, reward, termination, truncation, info ?
        action = Positions(action)
        self._done = (self._current_tick + 1) >= self._end_tick
        if self._done:
            action = Positions.Flat
        self._position_history.append(action)

        self._update_profit(action)
        if self._position != action:
            self._last_trade_tick = self._current_tick

        self._current_tick += 1
        reward = self._calculate_reward(action)
        self._total_reward += reward

        self._position = action

        return (
            self._get_observation(),
            reward,
            self._done,
            {}
        )

    def reset(self, *, seed=None):
        super().reset(seed=seed)

        self._start_tick = self.np_random.integers(
            self.window_size, len(self.prices) - self.episode_length)
        self._end_tick = self._start_tick + self.episode_length
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.Flat
        self._position_history = []
        self._total_reward = 0
        self._total_profit = 1
        self.history = []

        return self._get_observation()

    def close(self):
        pass

    def render(self):
        from matplotlib import pyplot as plt

        plt.style.use('seaborn')
        plt.figure(figsize=(25, 10), dpi=200)

        ticks = np.arange(self._start_tick, self._end_tick)
        prices = self.prices[ticks]
        ticks -= ticks.min()

        plt.plot(ticks, prices, 'b', alpha=0.3)
        plt.plot(ticks, prices, 'b.', alpha=0.3)

        for tick, position in zip(ticks, self._position_history):
            if position == Positions.Short:
                plt.plot([tick], [prices[tick]], 'ro', alpha=0.9)
            elif position == Positions.Flat:
                plt.plot([tick], [prices[tick]], 'bo', alpha=0.3)
            elif position == Positions.Long:
                plt.plot([tick], [prices[tick]], 'go', alpha=0.9)

        info = 'total profit: {:.3f}; idx_start: {};'.format(
            self._total_profit,
            self._start_tick
        )
        plt.title(info, fontsize=20)
        plt.xlabel('candle #')
        plt.ylabel('stock close price')

        plt.show()

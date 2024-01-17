from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd

from .base_env import BaseTradingEnv, Position


class TradingEnv(BaseTradingEnv):
    """
    It's a gymnasium environment that takes a dataframe of stock prices
    and allows you to trade on it
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        max_episode_steps: int = 24 * 14,
        window_size: int = 20,
        comission_fee: float = 7e-4,
        process_data: Callable = None,
        reward_mode: str = "step",
    ) -> None:
        self._setup_dataframe(df)
        self.window_size = window_size

        self.prices, self.signal_features = (
            process_data(self) if process_data else self._process_data()
        )
        if self.prices.shape[0] != self.signal_features.shape[0]:
            raise ValueError("signal_features and prices have different shapes")

        super().__init__(
            df=df, max_episode_steps=max_episode_steps, window_size=window_size,
            comission_fee=comission_fee, reward_mode=reward_mode
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

    def _get_features(self) -> np.ndarray:
        return self.signal_features[self._current_tick]

    def reset(self, idx_start=None, **kwargs) -> Tuple[Any, Dict]:
        super(BaseTradingEnv, self).reset(**kwargs)

        self._start_tick = self.np_random.integers(
            self.window_size, len(self.prices) - self.max_episode_steps
        ) if idx_start is None else idx_start
        self._end_tick = self._start_tick + self.max_episode_steps
        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Position.FLAT
        self._old_position = None
        self._position_history = []
        self._total_reward = 0
        self._total_profit = 1
        self._old_profit = None

        self._map_optimal_actions()

        return self._get_observation(), {}


if __name__ == '__main__':
    import gym_trading
    env = TradingEnv(df=gym_trading.datasets.BITCOIN_USD_1H, comission_fee=0.001, max_episode_steps=350)
    env.reset()
    print(env.get_max_profit())
    done_ = False
    while not done_:
        # action_ = env.action_space.sample()
        action_ = env.get_optimal_action()
        _, _, done_, _, _, = env.step(action_)
    print(env._total_profit)
    print(env._start_tick)
    print(env._total_reward)
    # env.render()
        
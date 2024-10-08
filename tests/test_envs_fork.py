import datetime

import numpy
import pandas
import pytest
from gymnasium.utils.env_checker import check_env

import gym_trading
from gym_trading.envs import Position, TradingEnv, TradingEnv2, TradingEnv3
from gym_trading.envs.base_env import BaseTradingEnv


@pytest.fixture()
def constant_price() -> pandas.DataFrame:
    """
    It creates a DataFrame with 10,000 rows, each row having a constant price
    :return: A dataframe with a constant price.
    """
    size = 20_000
    df = pandas.DataFrame()
    df[["open", "high", "low", "close"]] = numpy.ones((size, 4), dtype=numpy.float64)
    df["date"] = [
        datetime.datetime(2000, 1, 1, 0, 0) + i * datetime.timedelta(hours=1)
        for i in range(size)
    ]
    return df


@pytest.fixture()
def linear_price() -> pandas.DataFrame:
    """
    It creates a DataFrame with a linear price series
    :return: A dataframe with a linear price series.
    """
    size = 20_000
    price_start = 100
    price_end = 200
    df = pandas.DataFrame()
    df[["open", "high", "low", "close"]] = numpy.array(
        (
            numpy.linspace(price_start, price_end, size, dtype=numpy.float64),
            numpy.linspace(price_start, price_end, size, dtype=numpy.float64),
            numpy.linspace(price_start, price_end, size, dtype=numpy.float64),
            numpy.linspace(price_start, price_end, size, dtype=numpy.float64),
        )
    ).T
    df["date"] = [
        datetime.datetime(2000, 1, 1, 0, 0) + i * datetime.timedelta(hours=1)
        for i in range(size)
    ]
    return df


class TestBaseFunctionality:
    @pytest.fixture(params=[TradingEnv, TradingEnv2, TradingEnv3])
    def instance(self, request) -> BaseTradingEnv:
        return request.param

    def test_env_is_created_with_default_parameters(
        self,
        instance: BaseTradingEnv,
     ) -> None:
        """
        It creates an environment with default parameters
        """
        df = gym_trading.datasets.BITCOIN_USD_1H
        instance(df=df)
        instance(df=df.reset_index())
        instance(df=df.reset_index(drop=True))
        instance(df=df[df.columns[::-1]].reset_index(drop=True))
    
    def test_env_follows_gym_api(self, instance: BaseTradingEnv) -> None:
        """
        It checks that the environment follows the Gym API
        """
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)
        check_env(env, skip_render_check=True)

    def test_reset_refreshes_variables(self, instance: BaseTradingEnv) -> None:
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
        env.reset()

        assert env._position == Position.FLAT
        assert numpy.isclose(env._total_profit, 1.0)
        assert numpy.isclose(env._total_reward, 0.0)

    def test_buying_changes_position_to_LONG(self, instance: BaseTradingEnv) -> None:
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)
        env.reset()
        env.step(Position.LONG)
        assert env._position == Position.LONG

    def test_position_is_correct_after_many_steps(self, instance: BaseTradingEnv) -> None:
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)
        env.reset()
        env.step(Position.SHORT)
        env.step(Position.FLAT)
        env.step(Position.LONG)
        env.step(Position.SHORT)
        assert env._position == Position.SHORT
    
    def test_reward_depends_on_comission(self, instance: BaseTradingEnv) -> None:
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)
        for _ in range(10):
            env.reset()
            env.price.loc[env.price.index] = 1
            env.comission_fee = 0.01
            for _ in range(env._idx_last - env._idx_first - 1):
                pos = env._position.value
                action = env.action_space.sample()
                _, r, _, _, _ = env.step(action)
                assert (action == pos) == (r == 0)

    def test_reward_correlates_with_price_movement(self, instance: BaseTradingEnv) -> None:
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)

        # price moves up ==> reward for Long is positive
        env.reset()
        env.price.loc[env.price.index] = numpy.arange(100, 100 + len(env.price))
        env.comission_fee = 0.
        for _ in range(env._idx_last - env._idx_first - 1):
            action = env.action_space.sample()
            _, r, _, _, _ = env.step(action)
            assert numpy.sign(r) == numpy.sign(action - 1)
        
        # price moves down ==> reward for Short is positive
        env.reset()
        env.price.loc[env.price.index] = numpy.arange(100, 100 + len(env.price))[::-1]
        env.comission_fee = 0.
        for _ in range(env._idx_last - env._idx_first - 1):
            action = env.action_space.sample()
            _, r, _, _, _ = env.step(action)
            assert numpy.sign(r) == numpy.sign(1 - action)
    
    def test_optimal_actions_give_max_profit(self, instance: BaseTradingEnv) -> None:
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)

        for _ in range(30):
            env.comission_fee = numpy.random.uniform(0, 0.01)
            env.reset()
            done = False
            while not done:
                action = env.get_optimal_action()
                _, _, term, trunc, _ = env.step(action)
                done = (term or trunc)
            assert numpy.isclose(env._total_profit, env.get_max_profit())

    def test_observation_price_change_is_tracked_properly(self, instance: BaseTradingEnv) -> None:
        env = instance(df=gym_trading.datasets.BITCOIN_USD_1H)

        env.reset()
        indices = env.price.index[env._idx_first: env._idx_last + 1]
        # indices = env._indices[env._idx_first: env._idx_last + 1]
        env.price.loc[indices] = numpy.arange(100, 100 + len(indices))
        
        p0 = env.price.iloc[env._idx_now]
        assert p0 == 100.
        for _ in range(10):
            obs, _, _, _, _ = env.step(2)
        p1 = env.price.iloc[env._idx_now]
        assert p1 == 110
        assert numpy.isclose(
            obs['price_change'],
            numpy.arctan(100 * numpy.log(p1 / p0))
        )

        for _ in range(10):
            obs, _, _, _, _ = env.step(1)
        p2 = env.price.iloc[env._idx_now]
        assert p2 == 120
        assert numpy.isclose(
            obs['price_change'],
            numpy.arctan(100 * numpy.log(p2 / p1))
        )

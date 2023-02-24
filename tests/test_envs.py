import datetime
from pdb import line_prefix

import numpy
import pandas
import pytest
from gymnasium.utils.env_checker import check_env

import gym_trading
from gym_trading.envs import Position, TradingEnv


@pytest.fixture()
def constant_price() -> pandas.DataFrame:
    """
    It creates a DataFrame with 10,000 rows, each row having a constant price
    :return: A dataframe with a constant price.
    """
    size = 10_000
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
    size = 10_000
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


def test_env_is_created_with_default_parameters(
    constant_price: pandas.DataFrame,
    linear_price: pandas.DataFrame,
) -> None:
    """
    It creates an environment with default parameters
    """
    TradingEnv(df=gym_trading.datasets.BITCOIN_USD_1H)
    TradingEnv(df=constant_price)
    TradingEnv(df=linear_price)


class TestGymCompatability:
    """Collection of tests to verify that TradingEnv follows Gym API"""

    env = TradingEnv(df=gym_trading.datasets.BITCOIN_USD_1H)

    def test_env_follows_gym_api(self) -> None:
        """
        It checks that the environment follows the Gym API
        """
        check_env(self.env, skip_render_check=True)


class TestInnerLogic:
    """Colloction of tests to verify that Trading env emulates trades correctly"""

    def test_reset_refreshes_variables(self) -> None:
        env = TradingEnv(df=gym_trading.datasets.BITCOIN_USD_1H)
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
        env.reset()

        assert env._position == Position.Flat
        assert numpy.isclose(env._total_profit, 1.0)
        assert numpy.isclose(env._total_reward, 0.0)

    def test_buying_changes_position_to_long(self) -> None:
        env = TradingEnv(df=gym_trading.datasets.BITCOIN_USD_1H)
        env.reset()
        env.step(Position.Long)
        assert env._position == Position.Long

    def test_position_is_correct_after_many_steps(self) -> None:
        env = TradingEnv(df=gym_trading.datasets.BITCOIN_USD_1H)
        env.reset()
        env.step(Position.Short)
        env.step(Position.Flat)
        env.step(Position.Long)
        env.step(Position.Short)
        assert env._position == Position.Short

    def test_reward_for_operation_is_negative_fee(self, constant_price) -> None:
        env = TradingEnv(df=constant_price)
        env.reset()
        _1, reward, *_2 = env.step(Position.Long)
        assert numpy.isclose(reward, -env._comission_fee)
        _1, reward, *_2 = env.step(Position.Short)
        assert numpy.isclose(reward, 2 * -env._comission_fee)

    def test_reward_correlates_with_price_change(self, linear_price) -> None:
        env = TradingEnv(df=linear_price)
        env.reset()

        env.step(Position.Long)
        _1, reward, *_2 = env.step(Position.Long)
        assert reward > 0

        env.step(Position.Short)
        _1, reward, *_2 = env.step(Position.Short)
        assert reward < 0

        env.step(Position.Flat)
        _1, reward, *_2 = env.step(Position.Flat)
        assert reward == 0

    def test_profit_for_neutral_trade_is_based_on_fee(self, constant_price) -> None:
        env = TradingEnv(df=constant_price, comission_fee=0.0007)
        env.reset()
        for _ in range(100):
            env.step(Position.Long)
        env.step(Position.Flat)

        expected_profit = (1 - env._comission_fee) / (1 + env._comission_fee)
        assert numpy.isclose(env._total_profit, expected_profit)

    def test_profit_for_buy_and_hold(self, linear_price) -> None:
        env = TradingEnv(df=linear_price)
        env.reset()
        old_price = env.prices[env._current_tick]
        for _ in range(100):
            env.step(Position.Long)
        new_price = env.prices[env._current_tick]  # assumes that buy/sell is instant
        env.step(Position.Flat)

        expected_profit = (
            (new_price / old_price)
            * (1 - env._comission_fee)
            / (1 + env._comission_fee)
        )
        assert numpy.isclose(env._total_profit, expected_profit)

    def test_env_creation_with_custom_preprocessor(self) -> None:
        """
        It creates a trading environment with a custom preprocessor
        """

        def get_features(env) -> numpy.ndarray:
            return numpy.zeros(1_000), numpy.zeros((1_000, 5))

        TradingEnv(
            df=gym_trading.datasets.BITCOIN_USD_1H,
            process_data=get_features,
        )

import datetime

import numpy
import pandas
import pytest
from gymnasium.utils.env_checker import check_env

import gym_trading
from gym_trading.envs import TradingEnv


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

    def test_env_creation_with_custom_preprocessor(self) -> None:
        """
        It creates a trading environment with a custom preprocessor
        """

        def get_features(df: pandas.DataFrame, index: int) -> numpy.ndarray:
            return numpy.zeros(19)

        TradingEnv(
            df=gym_trading.datasets.BITCOIN_USD_1H,
            preprocessor=get_features,
        )

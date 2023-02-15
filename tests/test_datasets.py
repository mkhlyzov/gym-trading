import gym_trading


def test_hourly_bitcoin_is_accessable() -> None:
    df = gym_trading.datasets.BITCOIN_USD_1H
    assert not df.empty

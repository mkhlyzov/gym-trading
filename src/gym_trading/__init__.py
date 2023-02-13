from copy import deepcopy

from gym.envs.registration import register

from . import datasets
from . import envs

register(
    id='gym_trading/Trading-v0',
    entry_point='gym_trading.envs:TradingEnv',
    kwargs={
        'df': deepcopy(datasets.BITCOIN_USD_1H),
        'comission_fee': 7e-4,
        'window_size': 20,
        'episode_length': 24 * 14
    }
)

import random
from typing import Any, Dict, List, Tuple

from gym_trading.envs.base_env import BaseTradingEnv


class MixedEnv:
    """
    Wrapper that manages multiple gym_tarding environments.
    """
    def __init__(self, envs: List[BaseTradingEnv]) -> None:
        self.envs = envs
        self.current_env = self.envs[0]

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        self.current_env = random.choice(self.envs)
        return self.current_env.reset(**kwargs)
    
    def __getattr__(self, __name: str) -> Any:
        if hasattr(self.current_env, __name):
            return getattr(self.current_env, __name)
        raise AttributeError(f"'{self}' has no attribute '{__name}'")

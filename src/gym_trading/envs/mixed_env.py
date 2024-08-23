import random
from typing import Any, Dict, List, SupportsFloat, Tuple

import gymnasium

from gym_trading.envs.base_env import BaseTradingEnv


class MixedEnv(gymnasium.Env):
    """
    Wrapper-class that manages multiple gym_tarding environments.
    """
    def __init__(self, envs: List[BaseTradingEnv]) -> None:
        self.envs = envs
        self.weights = [len(env.df) for env in envs]
        self.current_env = self.envs[0]

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        self.current_env = random.choices(self.envs, self.weights)
        return self.current_env.reset(**kwargs)
    
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        return self.current_env.step(action)
    
    def close(self) -> None:
        for env in self.envs:
            env.close()

    def render(self) -> None:
        return self.current_env.render()
    
    def __getattr__(self, __name: str) -> Any:
        if hasattr(self.current_env, __name):
            return getattr(self.current_env, __name)
        raise AttributeError(f"'{self}' has no attribute '{__name}'")

from collections import deque

import numpy as np
import gymnasium


class ModifyReward(gymnasium.RewardWrapper):
    """ModifyReward wrapper for TradingEnv
    scale:      float, Multiply reward by this factor.
    discount:   float, Multiply NEGATIVE reward by this factor.
    clip:       float, Clips reward to interravl [-clip, clip].
                Clipping is applied after scaling but before discounting.

    Overall order:  R = discount( clip( scale(R) ) )
    """
    def __init__(
        self, env, *, scale: float=1., discount: float=1., clip: float|None=None
    ):
        super().__init__(env)
        self.scale = scale
        self.discount = discount
        self.clip = clip
        if discount < 0:
            raise ValueError("Discount factor must be non-negative")
        if clip is not None and clip <= 0:
            raise ValueError("Clip value must be positive")

    def reward(self, r):
        r *= self.scale

        if self.clip is not None:
            r = np.clip(r, -self.clip, self.clip)
        
        if r < 0:
            r *= self.discount

        return r


class StackObservations(gymnasium.ObservationWrapper):
    """StackObservations wrapper for TradingEnv
    n:      int, Number of observations to stack
    """
    def __init__(self, env, n: int):
        super(StackObservations, self).__init__(env)
        if n <= 0:
            raise ValueError(f"Can only stack poitive number of observations: {n}")
        self.n = n
        self.observation_space = self._modify_observation_space(env.observation_space)
        self.observation_buffer = {key: deque(maxlen=n) for key in self.observation_space.spaces.keys()}

    def _modify_observation_space(self, original_observation_space):
        # Modify each Box in the observation space to have shape (n * original_shape,)
        new_observation_space = {}
        for key, space in original_observation_space.spaces.items():
            original_shape = space.shape
            new_shape = (self.n * original_shape[0],)
            new_observation_space[key] = gymnasium.spaces.Box(
                low=np.repeat(space.low, self.n),
                high=np.repeat(space.high, self.n),
                shape=new_shape,
                dtype=space.dtype
            )
        return gymnasium.spaces.Dict(new_observation_space)

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        # Clear the buffer and initialize with the first observation
        for key in self.observation_buffer:
            self.observation_buffer[key].clear()
            for _ in range(self.n):
                self.observation_buffer[key].append(observation[key])
        return self._get_stacked_observation(), {}

    def observation(self, observation):
        # Add the new observation to the buffer
        for key in observation:
            self.observation_buffer[key].append(observation[key])
        return self._get_stacked_observation()

    def _get_stacked_observation(self):
        # Stack the observations for each key
        stacked_observation = {}
        for key, buffer in self.observation_buffer.items():
            stacked_observation[key] = np.concatenate(buffer, axis=0)
        return stacked_observation
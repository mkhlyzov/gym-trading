import random
import time

import numpy
import pandas

from gym_trading.envs import TradingEnv, TradingEnv2, TradingEnv3
from gym_trading.envs.base_env import BaseTradingEnv


def play_one_random_episode_and_time_it(env: BaseTradingEnv) -> float:
    """
    This function plays one episode randomly and measures how much time
    the whole process took.
    """
    time_elapsed = time.perf_counter()
    env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        # _, _, terminated, truncated, _ = env.step(env.get_optimal_action())
    time_elapsed = time.perf_counter() - time_elapsed

    return time_elapsed


def time_given_envs(
    envs: list[BaseTradingEnv],
    max_episodes: int = float("inf"),
    max_duration: float = float("inf"),
) -> None:
    """
    envs: list of environments (objects)
    max_duration: for how long this function should run (in seconds)

    This function plays many episodes on each Env and times each single episode.
    After that it prints average episode duration as well as standart diviation.
    Episodes are being run in random order to limit the influence of
    "cold/hot" cpu factor.
    """
    if max_episodes == float("inf") and max_duration == float("inf"):
        raise ValueError("Limits are poorly defined.")

    stats = {env: [] for env in envs}
    episodes_played, time_elapsed = 0, 0

    while episodes_played < max_episodes and time_elapsed < max_duration:
        env = random.choice(envs)
        t: float = play_one_random_episode_and_time_it(env)
        stats[env].append(t)

        episodes_played += 1
        time_elapsed += t

    for key, value in stats.items():
        # print(f"{repr(key): max(map(lambda x: len(repr(x)), env_fns))}")
        print(f"{repr(key)}")
        mean = numpy.mean(value)
        std = numpy.std(value) if len(value) > 1 else None
        if std is not None:
            # some numba.jit calls may significantly slow down first episode
            value = [v for v in value if abs(v - mean) < 3.5 * std]
            mean = numpy.mean(value)
            std = numpy.std(value) if len(value) > 1 else None
        print(f"average episode duration: {mean:.3f} ± {std:.3f} s          n={len(value)}")
        print()


def benchmark_envs(
    fname = "~/Dev/Datasets/cryptoarchive_close/BTCUSDT.csv"
) -> None:
    """
    This function is responsible for creating Env instances.
    After creating all instances it delegates to 'time_given_envs'
    """
    df = pandas.read_csv(fname)
    args = dict(df=df, max_episode_steps=350, window_size=50, comission_fee=0.001, reward_mode='optimal_action')
    args_to_print = {k: v for k, v in args.items() if k != 'df'}
    # printing the DataFrame itself only takes space w/o giving any important info

    print("==================================================")
    print("Warming up...")
    t0 = time.perf_counter()
    envs = [
        TradingEnv(**args),
        TradingEnv2(**args, std_threshold=0.0040),
        TradingEnv3(**args, std_threshold=0.0040),
    ]

    print(f"Warm up took: {time.perf_counter() - t0:.3f} seconds.")
    print("==================================================")
    print("Starting speed tests for envs")
    print(f"env args: {args_to_print}")
    print("==================================================")
    t0 = time.perf_counter()
    time_given_envs(
        envs,
        # max_episodes=100,
        max_duration=20.,   # run for 20 seconds
    )
    print("==================================================")
    print(f"Time taken: {time.perf_counter() - t0:.3f} seconds.")
    print("==================================================")


if __name__ == "__main__":
    benchmark_envs()

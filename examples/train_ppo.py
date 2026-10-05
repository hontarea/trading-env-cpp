"""Minimal end-to-end example: train PPO on a CSV and evaluate on held-out bars.

    python data/generate_data.py
    python examples/train_ppo.py data/synthetic_data.csv --timesteps 100000

For the full validation experiment (controls, cost sweep, seeds) see experiments/.
"""

import argparse
import math

import gymnasium as gym
from stable_baselines3 import PPO

import trading_env as te


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--commission-bp", type=float, default=1.0)
    args = parser.parse_args()

    data = te.MarketDataLoader.load_csv(args.csv)
    split = int(len(data) * (1.0 - args.test_fraction))
    common = dict(commission_rate=args.commission_bp * 1e-4, norm_range=(0, split))

    train_env = te.TradingGymEnv(data, bar_range=(0, split), episode_length=512, **common)
    # Per-bar log returns are tiny; scaling the reward helps PPO's value function.
    train_env = gym.wrappers.TransformReward(train_env, lambda r: 100.0 * r)
    model = PPO("MlpPolicy", train_env, gamma=0.95, gae_lambda=0.9, batch_size=256, ent_coef=0.01,
                seed=0, device="cpu", verbose=1)
    model.learn(total_timesteps=args.timesteps)

    test_env = te.TradingGymEnv(data, bar_range=(split, len(data)), **common)
    obs, _ = test_env.reset()
    log_return, done = 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        log_return += reward
        done = terminated or truncated

    print(f"Test bars {split}-{len(data)}: net return {math.expm1(log_return):+.2%}, "
          f"final equity {info['equity']:.2f}")


if __name__ == "__main__":
    main()

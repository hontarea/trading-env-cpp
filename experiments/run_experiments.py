"""Validation experiment: does PPO learn to trade a forecast through this environment?

Arms (all on the same synthetic price path, see data/generate_data.py):
  forecast  - the agent observes the true forecast (IC = 0.05);
  zero      - the forecast column is replaced by zeros;
  shuffled  - the forecast column is randomly permuted (same marginal, no information).
The forecast arm is trained at several commission levels; the two controls at one level.
Every run trains PPO on random 512-bar episodes from the training range and is evaluated
deterministically on the held-out test range. Rule-based baselines need no training.

Each training job runs in its own process with a single BLAS/torch thread. One pilot job runs
first; its peak memory decides how many jobs can run in parallel (capped by --workers).

Usage:
    python experiments/run_experiments.py                 # full grid, writes experiments/results
    python experiments/run_experiments.py --timesteps 20000 --seeds 1 --out /tmp/smoke
"""

from __future__ import annotations

import os

# Must be set before numpy / torch are imported (also inherited by spawned workers).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import math
import multiprocessing as mp
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))
from generate_data import generate  # noqa: E402

DATA = dict(n_bars=40_000, ic=0.05, sigma=0.005, phi=0.9, seed=0)
TRAIN_RANGE = (0, 30_000)
TEST_RANGE = (30_000, 40_000)
EPISODE_LENGTH = 512
BARS_PER_YEAR = 8760  # hourly bars
COSTS_BP = (0, 1, 5, 10)
CONTROL_COST_BP = 1
SHUFFLE_SEED = 12345
REWARD_SCALE = 100.0  # training only: per-bar log returns are O(1e-3)
CURVE_BIN = 10_000    # learning curves are averaged in bins of this many env steps

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.9,
    ent_coef=0.01,
)


@dataclass(frozen=True)
class Job:
    arm: str
    cost_bp: float
    seed: int
    timesteps: int

    @property
    def run_id(self) -> str:
        return f"{self.arm}_c{self.cost_bp:g}_s{self.seed}"


def market_data(arm: str):
    """Builds the dataset for an arm; prices are identical across arms."""
    import trading_env as te

    cols = generate(**DATA)
    forecast = cols["forecast"]
    if arm == "zero":
        forecast = np.zeros_like(forecast)
    elif arm == "shuffled":
        forecast = np.random.default_rng(SHUFFLE_SEED).permutation(forecast)
    elif arm != "forecast":
        raise ValueError(f"unknown arm {arm!r}")
    return te.MarketData.from_columns(
        cols["timestamp"], {"open": cols["open"], "close": cols["close"], "forecast": forecast}
    )


def make_env(data, cost_bp: float, bar_range, episode_length=None):
    import trading_env as te

    return te.TradingGymEnv(
        data,
        bar_range=bar_range,
        episode_length=episode_length,
        commission_rate=cost_bp * 1e-4,
        norm_range=TRAIN_RANGE,
    )


def evaluate(env, policy) -> dict:
    """Runs one deterministic episode over the env's whole range; policy(obs) -> action."""
    obs, _ = env.reset(seed=0)
    rewards, exposures, turnover = [], [], []
    prev_exposure, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
        rewards.append(reward)
        exposures.append(info["exposure"])
        turnover.append(abs(info["exposure"] - prev_exposure))
        prev_exposure = info["exposure"]
    rewards = np.asarray(rewards)
    exposures = np.asarray(exposures)
    log_return = float(rewards.sum())
    std = rewards.std()
    return {
        "net_return": math.expm1(log_return),
        "log_return": log_return,
        "sharpe": float(rewards.mean() / std * math.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0,
        "turnover": float(np.mean(turnover)),
        "mean_abs_exposure": float(np.abs(exposures).mean()),
        "frac_long": float(np.mean(exposures > 0.05)),
        "frac_short": float(np.mean(exposures < -0.05)),
    }


def run_job(job: Job) -> dict:
    """Trains and evaluates one PPO agent. Runs in a worker process."""
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    started = time.perf_counter()

    data = market_data(job.arm)
    train_env = make_env(data, job.cost_bp, TRAIN_RANGE, EPISODE_LENGTH)
    train_env = gym.wrappers.TransformReward(Monitor(train_env), lambda r: REWARD_SCALE * r)

    model = PPO("MlpPolicy", train_env, seed=job.seed, device="cpu", verbose=0, **PPO_KWARGS)
    model.learn(total_timesteps=job.timesteps)

    # Learning curve: mean training-episode log return per bin of env steps.
    monitor = train_env.env
    ends = np.cumsum(monitor.get_episode_lengths())
    returns = np.asarray(monitor.get_episode_rewards())
    bins = ends // CURVE_BIN
    curve = [
        {"run_id": job.run_id, "arm": job.arm, "cost_bp": job.cost_bp, "seed": job.seed,
         "timestep": int((b + 1) * CURVE_BIN), "episode_log_return": float(returns[bins == b].mean())}
        for b in np.unique(bins)
    ]

    test_env = make_env(data, job.cost_bp, TEST_RANGE)
    metrics = evaluate(test_env, lambda obs: int(model.predict(obs, deterministic=True)[0]))

    row = {
        "run_id": job.run_id, **asdict(job), **metrics,
        "wall_s": round(time.perf_counter() - started, 1),
        # ru_maxrss is in KiB on Linux: the peak resident memory of this worker process.
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    return {"row": row, "curve": curve}


def baselines() -> list[dict]:
    """Training-free reference policies on the test range."""
    rows = []
    data = market_data("forecast")
    for cost_bp in COSTS_BP:
        env = make_env(data, cost_bp, TEST_RANGE)
        forecast_col = env.observation_names.index("forecast")
        levels = list(env.exposure_levels)
        long_, flat, short = levels.index(1.0), levels.index(0.0), levels.index(-1.0)
        # Normalization is a monotone map, so the sign of the raw forecast is recovered from
        # the normalized value and the training-range mean/std.
        threshold = -env._mean[forecast_col] / env._std[forecast_col]
        policies = {
            "buy_and_hold": lambda obs: long_,
            "sign_forecast": lambda obs: long_ if obs[forecast_col] > threshold else short,
            "flat": lambda obs: flat,
        }
        for name, policy in policies.items():
            rows.append({"policy": name, "cost_bp": cost_bp, **evaluate(env, policy)})
    return rows


def available_memory_mb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6, help="max parallel training jobs")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--mem-fraction", type=float, default=0.6,
                        help="fraction of available memory the parallel jobs may use")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [Job("forecast", c, s, args.timesteps) for c in COSTS_BP for s in range(args.seeds)]
    jobs += [Job(arm, CONTROL_COST_BP, s, args.timesteps)
             for arm in ("zero", "shuffled") for s in range(args.seeds)]
    pilot = next(j for j in jobs if j.arm == "forecast" and j.cost_bp == CONTROL_COST_BP)
    jobs.remove(pilot)

    print(f"Baselines on test range {TEST_RANGE} ...", flush=True)
    write_csv(args.out / "baselines.csv", baselines())

    ctx = mp.get_context("spawn")
    rows, curves = [], []

    def collect(result: dict) -> None:
        row = result["row"]
        rows.append(row)
        curves.extend(result["curve"])
        print(f"[{len(rows):>2}/{len(jobs) + 1}] {row['run_id']:<20} test net return {row['net_return']:+8.3f}  "
              f"sharpe {row['sharpe']:+6.2f}  {row['wall_s']:>6.0f}s  peak {row['peak_rss_mb']:.0f} MB", flush=True)

    print(f"Pilot job {pilot.run_id} ({args.timesteps} steps) ...", flush=True)
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        collect(pool.submit(run_job, pilot).result())

    peak_mb = rows[0]["peak_rss_mb"]
    avail_mb = available_memory_mb()
    by_memory = max(1, int(args.mem_fraction * avail_mb // peak_mb))
    workers = min(args.workers, by_memory, len(jobs)) if jobs else 1
    print(f"Pilot peak RSS {peak_mb:.0f} MB, available memory {avail_mb:.0f} MB -> "
          f"memory allows {by_memory} jobs; running {len(jobs)} jobs on {workers} workers "
          f"(of {os.cpu_count()} cores)", flush=True)

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for future in as_completed([pool.submit(run_job, job) for job in jobs]):
            collect(future.result())

    rows.sort(key=lambda r: (r["arm"], r["cost_bp"], r["seed"]))
    write_csv(args.out / "runs.csv", rows)
    write_csv(args.out / "learning_curves.csv", curves)
    with open(args.out / "resources.txt", "w") as f:
        f.write(f"workers={workers} cores={os.cpu_count()} pilot_peak_rss_mb={peak_mb} "
                f"available_mb_at_start={avail_mb:.0f} timesteps={args.timesteps}\n")
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()

"""Synthetic OHLCV bars with a forecast of known predictive power.

The forecast f_t is a unit-variance AR(1) process (persistence ``phi``). The next bar's
log return is

    r_{t+1} = sigma * (ic * f_t + sqrt(1 - ic^2) * eps_{t+1}),    eps ~ N(0, 1)

so corr(f_t, r_{t+1}) = ic exactly: the forecast's information coefficient is a parameter.
Prices are a geometric random walk with no built-in mean reversion, and each bar opens at the
previous close (no artificial overnight gap). The only exploitable structure is the forecast.

Usage:
    python data/generate_data.py --n-bars 40000 --ic 0.05 --seed 0 --out data/synthetic_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BAR_SECONDS = 3600  # hourly bars


def generate(
    n_bars: int = 40_000,
    ic: float = 0.05,
    sigma: float = 0.005,
    phi: float = 0.9,
    seed: int = 0,
    start_price: float = 100.0,
) -> dict[str, np.ndarray]:
    """Returns ordered columns: timestamp, open, high, low, close, volume, forecast."""
    if not 0.0 <= ic < 1.0 or not 0.0 <= phi < 1.0 or sigma <= 0.0 or n_bars < 2:
        raise ValueError("need 0 <= ic < 1, 0 <= phi < 1, sigma > 0, n_bars >= 2")
    rng = np.random.default_rng(seed)

    forecast = np.empty(n_bars)
    forecast[0] = rng.standard_normal()
    innovations = rng.standard_normal(n_bars) * np.sqrt(1.0 - phi**2)
    for t in range(1, n_bars):
        forecast[t] = phi * forecast[t - 1] + innovations[t]

    noise = rng.standard_normal(n_bars)
    log_ret = np.zeros(n_bars)
    log_ret[1:] = sigma * (ic * forecast[:-1] + np.sqrt(1.0 - ic**2) * noise[1:])

    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0.0, sigma / 2, size=(2, n_bars)))
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    volume = np.round(rng.lognormal(mean=8.0, sigma=0.5, size=n_bars))

    return {
        "timestamp": np.arange(n_bars, dtype=np.int64) * BAR_SECONDS,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "forecast": forecast,
    }


def write_csv(columns: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    table = np.column_stack([columns[name] for name in names])
    fmt = ["%d"] + ["%.10g"] * (len(names) - 1)
    np.savetxt(path, table, delimiter=",", header=",".join(names), comments="", fmt=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-bars", type=int, default=40_000)
    parser.add_argument("--ic", type=float, default=0.05, help="corr(forecast_t, return_{t+1})")
    parser.add_argument("--sigma", type=float, default=0.005, help="per-bar return volatility")
    parser.add_argument("--phi", type=float, default=0.9, help="AR(1) persistence of the forecast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/synthetic_data.csv")
    args = parser.parse_args()

    columns = generate(args.n_bars, args.ic, args.sigma, args.phi, args.seed)
    write_csv(columns, args.out)
    realized_ic = np.corrcoef(columns["forecast"][:-1], np.diff(np.log(columns["close"])))[0, 1]
    print(f"Saved {args.n_bars} bars to {args.out} (realized IC {realized_ic:.4f})")


if __name__ == "__main__":
    main()

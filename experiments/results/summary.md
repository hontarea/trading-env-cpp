# Experiment results

Test range: bars 30,000-40,000 (held out). Mean ± std over seeds. Sharpe is annualized with 8,760 bars/year; turnover is mean |Δ exposure| per bar.

## PPO agents

| arm | commission (bp) | seeds | test log return | test Sharpe | turnover | mean abs exposure |
|---|---:|---:|---:|---:|---:|---:|
| forecast | 0 | 5 | +1.67 ± 0.21 | +3.30 ± 0.47 | +0.328 ± 0.046 | 0.91 |
| forecast | 1 | 5 | +1.47 ± 0.24 | +2.83 ± 0.45 | +0.220 ± 0.047 | 0.95 |
| forecast | 5 | 5 | +0.97 ± 0.26 | +1.82 ± 0.49 | +0.085 ± 0.024 | 0.99 |
| forecast | 10 | 5 | +0.88 ± 0.19 | +1.66 ± 0.35 | +0.065 ± 0.002 | 0.98 |
| zero | 1 | 5 | -0.32 ± 0.39 | -0.60 ± 0.78 | +0.132 ± 0.038 | 0.77 |
| shuffled | 1 | 5 | -0.34 ± 0.32 | -0.69 ± 0.58 | +0.228 ± 0.071 | 0.72 |

## Baselines (no training)

| policy | commission (bp) | test log return | test Sharpe | turnover |
|---|---:|---:|---:|---:|
| buy_and_hold | 0 | -0.25 | -0.47 | 0.000 |
| sign_forecast | 0 | +1.90 | +3.56 | 0.283 |
| flat | 0 | +0.00 | +0.00 | 0.000 |
| buy_and_hold | 1 | -0.25 | -0.47 | 0.000 |
| sign_forecast | 1 | +1.62 | +3.03 | 0.283 |
| flat | 1 | +0.00 | +0.00 | 0.000 |
| buy_and_hold | 5 | -0.25 | -0.48 | 0.000 |
| sign_forecast | 5 | +0.49 | +0.92 | 0.283 |
| flat | 5 | +0.00 | +0.00 | 0.000 |
| buy_and_hold | 10 | -0.25 | -0.48 | 0.000 |
| sign_forecast | 10 | -0.92 | -1.71 | 0.283 |
| flat | 10 | +0.00 | +0.00 | 0.000 |

## Compute

`workers=5 cores=20 pilot_peak_rss_mb=674.5 available_mb_at_start=6489 timesteps=300000`

| | mean | max |
|---|---:|---:|
| peak RSS per job (MB) | 678 | 687 |
| wall time per job (s) | 98 | 102 |

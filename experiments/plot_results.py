"""Turns experiments/results/*.csv into the README figures and summary.md.

Usage: python experiments/plot_results.py [--results experiments/results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Categorical slots 1-3 (validated all-pairs for colour-vision deficiencies), plus text inks.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ARM_COLORS = {"forecast": "#2a78d6", "zero": "#eb6834", "shuffled": "#1baf7a"}
ARM_LABELS = {"forecast": "PPO + forecast", "zero": "PPO + zero forecast", "shuffled": "PPO + shuffled forecast"}
# One-hue sequential ramp for the ordinal cost levels (light = cheap, dark = expensive).
COST_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def band(ax, df: pd.DataFrame, color: str, label: str, direct_label: bool = True) -> None:
    g = df.groupby("timestep")["episode_log_return"].agg(["mean", "std"]).reset_index()
    x = g["timestep"] / 1000
    ax.plot(x, g["mean"], color=color, linewidth=2, label=label)
    ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15, linewidth=0)
    if direct_label:
        ax.annotate(label, (x.iloc[-1], g["mean"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)


def plot_learning_curves(curves: pd.DataFrame, out: Path, control_cost: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, sharey=True)

    ax = axes[0]
    for arm in ("forecast", "zero", "shuffled"):
        sub = curves[(curves.arm == arm) & (curves.cost_bp == control_cost)]
        if not sub.empty:
            # The two controls end on top of each other; the legend identifies them.
            band(ax, sub, ARM_COLORS[arm], ARM_LABELS[arm], direct_label=arm == "forecast")
    ax.set_title(f"Forecast vs. controls ({control_cost:g} bp commission)", fontsize=11, loc="left")

    ax = axes[1]
    forecast = curves[curves.arm == "forecast"]
    for color, cost in zip(COST_RAMP, sorted(forecast.cost_bp.unique())):
        # Curves overlap here, so the legend carries identity instead of direct labels.
        band(ax, forecast[forecast.cost_bp == cost], color, f"{cost:g} bp", direct_label=False)
    ax.set_title("PPO + forecast by commission level", fontsize=11, loc="left")

    for ax in axes:
        style_axes(ax)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xlabel("training steps (thousands)")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=4,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18))
        ax.margins(x=0.02)
    axes[0].set_xlim(right=axes[0].get_xlim()[1] * 1.15)  # room for direct labels
    axes[0].set_ylabel("training episode log return\n(512 bars, mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_test_results(runs: pd.DataFrame, baselines: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    metrics = [("sharpe", "test Sharpe ratio (annualized)"), ("turnover", "mean |Δ exposure| per bar")]
    forecast = runs[runs.arm == "forecast"]
    agg = forecast.groupby("cost_bp")[[m for m, _ in metrics]].mean().reset_index()

    for ax, (metric, ylabel) in zip(axes, metrics):
        rule = baselines[baselines.policy == "sign_forecast"].sort_values("cost_bp")
        ax.plot(rule.cost_bp, rule[metric], color=INK_2, linewidth=1.5, linestyle="--",
                marker="o", markersize=5, label="sign(forecast) rule, no training")
        ax.scatter(forecast.cost_bp, forecast[metric], s=22, color=ARM_COLORS["forecast"], alpha=0.35,
                   linewidths=0, zorder=3)
        ax.plot(agg.cost_bp, agg[metric], color=ARM_COLORS["forecast"], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label="PPO + forecast (mean, seeds)",
                zorder=4)
        for arm in ("zero", "shuffled"):
            sub = runs[runs.arm == arm]
            if sub.empty:
                continue
            jitter = 0.25 if arm == "shuffled" else -0.25
            ax.scatter(sub.cost_bp + jitter, sub[metric], s=34, color=ARM_COLORS[arm], marker="D",
                       edgecolors=SURFACE, linewidths=1.5, label=ARM_LABELS[arm], zorder=5)
        if metric == "sharpe":
            bh = baselines[baselines.policy == "buy_and_hold"].sort_values("cost_bp")
            ax.plot(bh.cost_bp, bh[metric], color=INK_2, linewidth=1, linestyle=":", label="buy and hold")
            ax.axhline(0, color=INK_2, linewidth=0.8)
        style_axes(ax)
        ax.set_xlabel("commission (bp of traded notional)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(baselines.cost_bp.unique()))
    axes[0].set_title("Out-of-sample performance (held-out 10,000 bars)", fontsize=11, loc="left")
    axes[1].set_title("Trading activity", fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:+.{digits}f} ± {std:.{digits}f}"


def write_summary(runs: pd.DataFrame, baselines: pd.DataFrame, resources: str, out: Path) -> None:
    lines = ["# Experiment results", "",
             "Test range: bars 30,000-40,000 (held out). Mean ± std over seeds. "
             "Sharpe is annualized with 8,760 bars/year; turnover is mean |Δ exposure| per bar.", "",
             "## PPO agents", "",
             "| arm | commission (bp) | seeds | test log return | test Sharpe | turnover | mean abs exposure |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for (arm, cost), g in runs.groupby(["arm", "cost_bp"], sort=False):
        lines.append(f"| {arm} | {cost:g} | {len(g)} | {fmt(g.log_return.mean(), g.log_return.std())} | "
                     f"{fmt(g.sharpe.mean(), g.sharpe.std())} | {fmt(g.turnover.mean(), g.turnover.std(), 3)} | "
                     f"{g.mean_abs_exposure.mean():.2f} |")
    lines += ["", "## Baselines (no training)", "",
              "| policy | commission (bp) | test log return | test Sharpe | turnover |", "|---|---:|---:|---:|---:|"]
    for _, r in baselines.iterrows():
        lines.append(f"| {r.policy} | {r.cost_bp:g} | {r.log_return:+.2f} | {r.sharpe:+.2f} | {r.turnover:.3f} |")
    lines += ["", "## Compute", "",
              f"`{resources.strip()}`", "",
              "| | mean | max |", "|---|---:|---:|",
              f"| peak RSS per job (MB) | {runs.peak_rss_mb.mean():.0f} | {runs.peak_rss_mb.max():.0f} |",
              f"| wall time per job (s) | {runs.wall_s.mean():.0f} | {runs.wall_s.max():.0f} |", ""]
    out.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "experiments" / "results")
    parser.add_argument("--control-cost", type=float, default=1.0)
    args = parser.parse_args()

    runs = pd.read_csv(args.results / "runs.csv")
    runs["_order"] = runs.arm.map({"forecast": 0, "zero": 1, "shuffled": 2})
    runs = runs.sort_values(["_order", "cost_bp", "seed"]).drop(columns="_order")
    curves = pd.read_csv(args.results / "learning_curves.csv")
    # PPO rounds training up to whole rollouts; drop the partial bin past the step budget.
    curves = curves[curves.timestep <= runs.timesteps.max()]
    baselines = pd.read_csv(args.results / "baselines.csv")
    resources = (args.results / "resources.txt").read_text()

    plot_learning_curves(curves, args.results / "learning_curves.png", args.control_cost)
    plot_test_results(runs, baselines, args.results / "test_results.png")
    write_summary(runs, baselines, resources, args.results / "summary.md")
    print(f"Wrote figures and summary.md to {args.results}")


if __name__ == "__main__":
    main()

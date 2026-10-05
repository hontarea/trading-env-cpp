"""Gymnasium wrapper around the C++ TradingEnv."""

from __future__ import annotations

import os
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import _core


class TradingGymEnv(gym.Env):
    """Gymnasium environment for trading a single asset on a forecast.

    The agent observes bar ``t`` (close and features), picks one of ``exposure_levels``,
    the order fills at the open of bar ``t+1`` and the reward is the log equity growth up to
    the close of bar ``t+1``, net of spread, impact and commission.

    Args:
        data: Path to a CSV file or a ``MarketData`` instance.
        bar_range: ``(start, end)`` bars that episodes are drawn from (end exclusive).
            Defaults to the whole dataset.
        episode_length: Steps per episode. ``None`` runs one episode over all of
            ``bar_range``; otherwise each reset picks a random start (via ``np_random``).
        exposure_levels: Target exposures for the discrete actions, each in ``[-1, 1]``
            and scaled by ``max_leverage``.
        max_leverage: Exposure (position value / equity) reached at action level 1.
        half_spread, impact_coef, commission_rate: Execution cost parameters, as
            fractions (1 bp = 1e-4). See ``ExecutionModel``.
        obs_features: Data columns added to the observation. ``None`` uses
            ``("forecast",)`` when the data has that column, else nothing.
        obs_lookback: Number of bars of history per observation feature.
        normalize: Z-score the market part of the observation.
        norm_range: Bars used to fit the normalization statistics. Defaults to
            ``bar_range``. For an evaluation env pass the *training* range, so that no
            statistics of the test data leak into the observations.
        initial_cash: Starting cash per episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: str | os.PathLike | _core.MarketData,
        *,
        bar_range: tuple[int, int] | None = None,
        episode_length: int | None = None,
        exposure_levels: Sequence[float] = (-1.0, 0.0, 1.0),
        max_leverage: float = 1.0,
        half_spread: float = 0.0,
        impact_coef: float = 0.0,
        commission_rate: float = 0.0,
        obs_features: Sequence[str] | None = None,
        obs_lookback: int = 1,
        normalize: bool = True,
        norm_range: tuple[int, int] | None = None,
        initial_cash: float = 10_000.0,
    ):
        super().__init__()

        if isinstance(data, (str, os.PathLike)):
            data = _core.MarketDataLoader.load_csv(os.fspath(data))
        self.data = data

        self.bar_range = (0, len(data)) if bar_range is None else (int(bar_range[0]), int(bar_range[1]))
        start, end = self.bar_range
        if not 0 <= start < end <= len(data) or end - start < 2:
            raise _core.ConfigError(f"bar_range {self.bar_range} is invalid for {len(data)} bars")
        if episode_length is not None and not 1 <= episode_length < end - start:
            raise _core.ConfigError(
                f"episode_length must be in [1, {end - start - 1}] for bar_range {self.bar_range}"
            )
        self.episode_length = episode_length

        self.exposure_levels = np.asarray(exposure_levels, dtype=np.float64)
        if self.exposure_levels.ndim != 1 or len(self.exposure_levels) < 2:
            raise _core.ConfigError("exposure_levels needs at least two levels")
        if np.any(np.abs(self.exposure_levels) > 1.0):
            raise _core.ConfigError("exposure_levels must lie in [-1, 1]")

        if obs_features is None:
            obs_features = (_core.FEATURE_FORECAST,) if data.has_feature(_core.FEATURE_FORECAST) else ()

        config = _core.EnvConfig()
        config.initial_cash = float(initial_cash)
        config.start_index, config.end_index = self.bar_range
        config.max_leverage = float(max_leverage)
        config.obs_features = list(obs_features)
        config.obs_lookback = int(obs_lookback)

        self.execution = _core.ExecutionModel(half_spread, impact_coef, commission_rate)
        self._env = _core.TradingEnv(data, self.execution, _core.log_return_reward(), config)
        self.observation_names = list(self._env.observation_names())

        n_obs = self._env.observation_size
        self._n_market = self._env.market_feature_count
        self._mean = np.zeros(n_obs)
        self._std = np.ones(n_obs)
        if normalize:
            self.norm_range = self.bar_range if norm_range is None else tuple(norm_range)
            self._fit_normalization(*self.norm_range)
        else:
            self.norm_range = None

        self.action_space = spaces.Discrete(len(self.exposure_levels))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

    def _fit_normalization(self, start: int, end: int) -> None:
        """Mean/std of the market features over bars [start, end) only."""
        if not 0 <= start < end <= len(self.data):
            raise _core.ConfigError(f"norm_range {(start, end)} is invalid for {len(self.data)} bars")
        # Bar `start` has no history, so its return features are 0 by construction; skip it.
        first = start + 1 if end - start > 1 else start
        features = np.stack([self._env.market_features(t) for t in range(first, end)])
        self._mean[: self._n_market] = features.mean(axis=0)
        self._std[: self._n_market] = np.maximum(features.std(axis=0), 1e-8)

    def _obs(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self._mean) / self._std).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        start, end = self.bar_range
        if self.episode_length is not None:
            n_bars = self.episode_length + 1
            start = int(self.np_random.integers(start, end - n_bars + 1))
            end = start + n_bars
        raw = self._env.reset(start, end)
        return self._obs(raw), {"step": start}

    def step(self, action):
        target = float(self.exposure_levels[int(action)])
        result = self._env.step(target)
        info = result.info.to_dict()
        info["step"] = self._env.current_step()
        info["target_exposure"] = target
        return self._obs(result.observation), result.reward, result.terminated, result.truncated, info

    @property
    def portfolio(self):
        return self._env.portfolio()

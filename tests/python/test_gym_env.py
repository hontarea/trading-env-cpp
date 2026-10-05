import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import trading_env as te
from conftest import make_market_data


def test_passes_gymnasium_check_env(market_data):
    env = te.TradingGymEnv(market_data, episode_length=50, commission_rate=1e-4)
    check_env(env, skip_render_check=True)


def test_loads_csv_path(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("timestamp,open,close\n1,100,101\n2,101,102\n3,102,103\n")
    env = te.TradingGymEnv(path)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape == (5,)
    assert env.observation_names[-3:] == ["exposure", "log_equity_growth", "episode_frac_remaining"]


def rollout(env, seed, n_steps=200):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    trace = [obs]
    for _ in range(n_steps):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        trace.append(np.append(obs, reward))
        if terminated or truncated:
            obs, _ = env.reset()
            trace.append(obs)
    return trace


def test_same_seed_gives_identical_trajectories(market_data):
    make = lambda: te.TradingGymEnv(market_data, episode_length=30, commission_rate=5e-4)
    a, b = rollout(make(), seed=7), rollout(make(), seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = rollout(make(), seed=8)
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_full_range_episode_ends_in_truncation(market_data):
    env = te.TradingGymEnv(market_data, bar_range=(100, 110))
    env.reset()
    for i in range(9):
        obs, reward, terminated, truncated, info = env.step(2)
        assert not terminated
        assert truncated == (i == 8)
    assert np.all(np.isfinite(obs))
    assert info["step"] == 109


def test_info_reports_execution(market_data):
    env = te.TradingGymEnv(market_data, commission_rate=1e-3)
    env.reset()
    _, _, _, _, info = env.step(2)  # level +1: full long
    notional = info["units_traded"] * info["fill_price"]
    assert info["commission"] == pytest.approx(1e-3 * notional)
    assert info["target_exposure"] == 1.0
    assert 0.9 < info["exposure"] < 1.1
    assert info["bankrupt"] is False


def test_rewards_sum_to_log_equity_growth(market_data):
    env = te.TradingGymEnv(market_data, half_spread=2e-4, commission_rate=5e-4)
    env.reset()
    total, done = 0.0, False
    rng = np.random.default_rng(0)
    while not done:
        _, reward, terminated, truncated, info = env.step(int(rng.integers(3)))
        total += reward
        done = terminated or truncated
    assert total == pytest.approx(math.log(info["equity"] / 10_000.0), abs=1e-9)


def test_normalization_uses_only_norm_range():
    data = make_market_data(n_bars=400)
    # Same training range, different test data -> identical normalization statistics.
    other = make_market_data(n_bars=400, seed=1)
    train = (0, 200)
    a = te.TradingGymEnv(data, bar_range=(200, 400), norm_range=train)
    b = te.TradingGymEnv(data, bar_range=train)
    assert np.array_equal(a._mean, b._mean) and np.array_equal(a._std, b._std)
    c = te.TradingGymEnv(other, bar_range=train)
    assert not np.array_equal(a._mean, c._mean)
    # Normalized market features on the training range are ~zero-mean, unit-variance.
    obs = np.stack([b._obs(b._env.market_features(t).tolist() + [0, 0, 0]) for t in range(1, 200)])
    n = b._n_market
    assert np.allclose(obs[:, :n].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(obs[:, :n].std(axis=0), 1.0, atol=1e-4)


def test_exceptions_are_mapped(market_data, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,open,close\n1,100,1.5abc\n")
    with pytest.raises(te.DataLoadError):
        te.MarketDataLoader.load_csv(str(bad))
    with pytest.raises(te.ConfigError):
        te.TradingGymEnv(market_data, obs_features=["missing"])
    assert issubclass(te.DataLoadError, te.MarketSimError)

    core = te.TradingEnv(market_data, te.ExecutionModel(), te.log_return_reward(), te.EnvConfig())
    with pytest.raises(te.EpisodeError):
        core.step(0.0)
    core.reset()
    with pytest.raises(te.InvalidActionError):
        core.step(1.5)


def test_random_starts_cover_the_range(market_data):
    env = te.TradingGymEnv(market_data, bar_range=(0, 300), episode_length=50)
    starts = {env.reset(seed=s)[1]["step"] for s in range(50)}
    assert min(starts) >= 0 and max(starts) <= 300 - 51
    assert len(starts) > 20

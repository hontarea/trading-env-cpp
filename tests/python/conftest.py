import numpy as np
import pytest

import trading_env as te


def make_market_data(n_bars: int = 400, seed: int = 0) -> te.MarketData:
    rng = np.random.default_rng(seed)
    forecast = rng.standard_normal(n_bars)
    log_ret = 0.01 * (0.1 * np.roll(forecast, 1) + rng.standard_normal(n_bars))
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    return te.MarketData.from_columns(
        np.arange(n_bars, dtype=np.int64) * 3600,
        {"open": open_, "close": close, "forecast": forecast},
    )


@pytest.fixture
def market_data():
    return make_market_data()

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "Exceptions.h"
#include "helpers.h"

using test::costs;
using test::make_data;
using test::make_env;

namespace {

const std::vector<double> kFlat5{100, 100, 100, 100, 100};

}  // namespace

TEST(Env, ObservationAtTIgnoresFutureBars) {
    // Two datasets agree on bars 0..4 and differ afterwards. Observations up to t = 4 must match.
    const std::vector<double> base{100, 101, 99, 102, 103};
    auto a_open = base, a_close = base, b_open = base, b_close = base;
    std::vector<double> fa{0.1, -0.2, 0.3, 0.4, -0.5}, fb = fa;
    for (int i = 0; i < 5; ++i) {
        a_open.push_back(110 + i); a_close.push_back(111 + i); fa.push_back(1.0);
        b_open.push_back(50 - i);  b_close.push_back(49 - i);  fb.push_back(-1.0);
    }
    EnvConfig config;
    config.obs_features = {"forecast"};
    config.obs_lookback = 2;
    auto env_a = make_env(make_data(a_open, a_close, {{"forecast", fa}}), costs(0.001, 0.01, 0.001), config);
    auto env_b = make_env(make_data(b_open, b_close, {{"forecast", fb}}), costs(0.001, 0.01, 0.001), config);

    EXPECT_EQ(env_a.reset(), env_b.reset());
    const double actions[] = {1.0, -0.5, 0.0, 1.0};
    for (double action : actions) {  // reaches t = 4
        auto ra = env_a.step(action);
        auto rb = env_b.step(action);
        EXPECT_EQ(ra.observation, rb.observation) << "t = " << env_a.current_step();
        EXPECT_EQ(ra.reward, rb.reward);
    }
    for (std::size_t t = 0; t <= 4; ++t) {
        EXPECT_EQ(env_a.market_features(t), env_b.market_features(t));
    }
}

TEST(Env, OrderExecutesAtNextOpen) {
    // close[0] = 100 is what the agent sees; the order must fill at open[1] = 120.
    auto env = make_env(make_data({100, 120, 120}, {100, 130, 130}));
    env.reset();
    auto r = env.step(1.0);
    EXPECT_DOUBLE_EQ(r.info.fill_price, 120.0);
    EXPECT_NEAR(r.info.equity, 10000.0 * 130.0 / 120.0, 1e-9);
}

TEST(Env, EndOfDataIsTruncationWithRealObservation) {
    EnvConfig config;
    config.obs_features = {"forecast"};
    auto env = make_env(make_data(kFlat5, kFlat5, {{"forecast", {1, 2, 3, 4, 5}}}), costs(), config);
    env.reset();
    StepResult r;
    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(env.done());
        r = env.step(0.0);
        EXPECT_FALSE(r.terminated);
        EXPECT_EQ(r.truncated, i == 3);
    }
    EXPECT_TRUE(env.done());
    ASSERT_EQ(r.observation.size(), env.observation_size());
    EXPECT_DOUBLE_EQ(r.observation[2], 5.0);       // forecast of the last bar, not zeroed
    EXPECT_DOUBLE_EQ(r.observation.back(), 0.0);   // no episode left
    EXPECT_THROW(env.step(0.0), EpisodeError);
}

TEST(Env, StepBeforeResetThrows) {
    auto env = make_env(make_data(kFlat5, kFlat5));
    EXPECT_THROW(env.step(0.0), EpisodeError);
}

TEST(Env, BankruptcyTerminatesWithFiniteReward) {
    EnvConfig config;
    config.max_leverage = 2.0;
    auto env = make_env(make_data({100, 100, 40, 40}, {100, 40, 40, 40}), costs(), config);
    env.reset();
    auto r = env.step(1.0);  // 2x long at 100, close at 40: equity = 10000 - 200 * 60 < 0
    EXPECT_TRUE(r.info.bankrupt);
    EXPECT_TRUE(r.terminated);
    EXPECT_FALSE(r.truncated);
    EXPECT_TRUE(std::isfinite(r.reward));
    EXPECT_NEAR(r.reward, std::log(RewardFunctions::MIN_EQUITY_RATIO), 1e-12);
    for (double v : r.observation) {
        EXPECT_TRUE(std::isfinite(v));
    }
    EXPECT_THROW(env.step(0.0), EpisodeError);
}

TEST(Env, RejectsInvalidActions) {
    auto env = make_env(make_data(kFlat5, kFlat5));
    env.reset();
    EXPECT_THROW(env.step(1.01), InvalidActionError);
    EXPECT_THROW(env.step(-2.0), InvalidActionError);
    EXPECT_THROW(env.step(std::numeric_limits<double>::quiet_NaN()), InvalidActionError);
    EXPECT_NO_THROW(env.step(-1.0));
}

TEST(Env, ConstructorValidation) {
    auto data = make_data(kFlat5, kFlat5);
    auto reward = RewardFunctions::log_return_reward();
    EXPECT_THROW(TradingEnv(nullptr, costs(), reward, {}), ConfigError);
    EXPECT_THROW(TradingEnv(data, nullptr, reward, {}), ConfigError);
    EXPECT_THROW(TradingEnv(data, costs(), nullptr, {}), ConfigError);

    EnvConfig bad;
    bad.obs_features = {"missing"};
    EXPECT_THROW(make_env(data, costs(), bad), ConfigError);
    bad = {};
    bad.start_index = 4;
    EXPECT_THROW(make_env(data, costs(), bad), ConfigError);
    bad = {};
    bad.end_index = 6;
    EXPECT_THROW(make_env(data, costs(), bad), ConfigError);
    bad = {};
    bad.max_leverage = 0.0;
    EXPECT_THROW(make_env(data, costs(), bad), ConfigError);
    bad = {};
    bad.initial_cash = -1.0;
    EXPECT_THROW(make_env(data, costs(), bad), ConfigError);

    auto no_close = std::make_shared<const MarketData>(
        MarketData::from_columns({0, 1}, {{"open", {1.0, 1.0}}}));
    EXPECT_THROW(make_env(no_close), ConfigError);
    EXPECT_THROW(make_env(make_data({100, 0}, {100, 100})), ConfigError);
}

TEST(Env, ObservationLayoutAndLookback) {
    EnvConfig config;
    config.obs_features = {"f"};
    config.obs_lookback = 3;
    auto env = make_env(make_data(kFlat5, {100, 110, 121, 121, 121}, {{"f", {10, 11, 12, 13, 14}}}),
                        costs(), config);
    const std::vector<std::string> expected_names{
        "log_return", "mean_log_return_5", "f", "f_lag1", "f_lag2",
        "exposure", "log_equity_growth", "episode_frac_remaining"};
    EXPECT_EQ(env.observation_names(), expected_names);
    EXPECT_EQ(env.observation_size(), expected_names.size());
    EXPECT_EQ(env.market_feature_count(), 5u);

    auto obs = env.reset(1, 5);
    EXPECT_NEAR(obs[0], std::log(1.1), 1e-12);
    EXPECT_NEAR(obs[1], std::log(1.1), 1e-12);   // only one return available at t = 1
    EXPECT_DOUBLE_EQ(obs[2], 11.0);
    EXPECT_DOUBLE_EQ(obs[3], 10.0);
    EXPECT_DOUBLE_EQ(obs[4], 10.0);              // lag before bar 0 repeats bar 0
    EXPECT_DOUBLE_EQ(obs[5], 0.0);
    EXPECT_DOUBLE_EQ(obs[6], 0.0);
    EXPECT_DOUBLE_EQ(obs[7], 1.0);

    auto mf = env.market_features(3);
    EXPECT_NEAR(mf[1], std::log(121.0 / 100.0) / 3.0, 1e-12);
}

TEST(Env, ResetSubRangeAndValidation) {
    auto env = make_env(make_data(kFlat5, kFlat5));
    env.reset(2, 4);
    EXPECT_EQ(env.current_step(), 2u);
    auto r = env.step(0.0);
    EXPECT_TRUE(r.truncated);
    EXPECT_THROW(env.reset(3, 4), ConfigError);  // a single bar cannot form a step
    EXPECT_THROW(env.reset(0, 6), ConfigError);
}

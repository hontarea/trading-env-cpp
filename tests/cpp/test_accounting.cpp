#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "Portfolio.h"
#include "helpers.h"

using test::costs;
using test::make_data;
using test::make_env;

TEST(Portfolio, LongRoundTripHasExactPnL) {
    Portfolio p(10000.0);
    p.apply_fill(10.0, 100.0, 1.0);    // buy 10 @ 100, fee 1
    p.mark_to_market(110.0);
    EXPECT_DOUBLE_EQ(p.equity(), 10000.0 - 1000.0 - 1.0 + 1100.0);
    p.apply_fill(-10.0, 110.0, 1.1);   // sell 10 @ 110, fee 1.1
    p.mark_to_market(110.0);
    EXPECT_DOUBLE_EQ(p.units(), 0.0);
    EXPECT_DOUBLE_EQ(p.cash(), 10000.0 + 10.0 * (110.0 - 100.0) - 1.0 - 1.1);
    EXPECT_DOUBLE_EQ(p.equity(), p.cash());
}

TEST(Portfolio, ShortGainsWhenPriceFalls) {
    // Short 1 unit at 100, price moves to 90: equity rises by exactly 10 minus costs.
    Portfolio p(10000.0);
    p.mark_to_market(100.0);
    p.apply_fill(-1.0, 100.0, 0.25);
    p.mark_to_market(90.0);
    EXPECT_DOUBLE_EQ(p.units(), -1.0);
    EXPECT_DOUBLE_EQ(p.cash(), 10100.0 - 0.25);
    EXPECT_DOUBLE_EQ(p.equity(), 10000.0 + 10.0 - 0.25);
    EXPECT_LT(p.exposure(), 0.0);
}

TEST(Env, LongRoundTripMatchesPriceRatio) {
    // Without costs, buying at open[1] and selling at open[2] returns open[2] / open[1].
    auto env = make_env(make_data({100, 100, 125, 130}, {100, 110, 120, 130}));
    env.reset();
    env.step(1.0);
    auto r = env.step(0.0);
    EXPECT_NEAR(r.info.equity, 10000.0 * 125.0 / 100.0, 1e-9);
    EXPECT_DOUBLE_EQ(env.portfolio().units(), 0.0);
}

TEST(Env, FullShortProfitsFromFall) {
    auto env = make_env(make_data({100, 100, 90}, {100, 100, 90}));
    env.reset();
    env.step(-1.0);                       // short 100 units at open[1] = 100
    auto r = env.step(0.0);               // cover at open[2] = 90
    EXPECT_NEAR(r.info.equity, 10000.0 + 100.0 * 10.0, 1e-9);
}

TEST(Env, EquityIdentityAndRewardSumHoldUnderCosts) {
    std::mt19937 rng(42);
    std::normal_distribution<double> ret(0.0, 0.02);
    std::uniform_real_distribution<double> action(-1.0, 1.0);

    std::vector<double> open, close;
    double price = 100.0;
    for (int i = 0; i < 300; ++i) {
        open.push_back(price * std::exp(ret(rng) * 0.3));
        price = open.back() * std::exp(ret(rng));
        close.push_back(price);
    }
    EnvConfig config;
    config.max_leverage = 1.5;
    auto env = make_env(make_data(open, close), costs(0.0005, 0.001, 0.0002), config);

    env.reset();
    double reward_sum = 0.0;
    bool done = false;
    while (!done) {
        auto r = env.step(action(rng));
        reward_sum += r.reward;
        done = r.terminated || r.truncated;
        const auto& p = env.portfolio();
        const double price_now = close[env.current_step()];
        EXPECT_NEAR(p.equity(), p.cash() + p.units() * price_now, 1e-9 * p.equity());
        EXPECT_NEAR(r.info.exposure, p.units() * price_now / p.equity(), 1e-12);
    }
    EXPECT_NEAR(reward_sum, std::log(env.portfolio().equity() / config.initial_cash), 1e-9);
}

TEST(Env, ExposureMatchesTargetAfterFill) {
    // With zero costs and no intra-bar move, exposure after the fill equals target * leverage.
    EnvConfig config;
    config.max_leverage = 2.0;
    auto env = make_env(make_data({100, 100, 100}, {100, 100, 100}), costs(), config);
    env.reset();
    auto r = env.step(0.5);
    EXPECT_NEAR(r.info.exposure, 1.0, 1e-12);
    EXPECT_NEAR(env.portfolio().units(), 100.0, 1e-9);
}

#include <gtest/gtest.h>

#include "Exceptions.h"
#include "ExecutionModel.h"
#include "helpers.h"

using test::costs;
using test::make_data;
using test::make_env;

TEST(ExecutionModel, BuysFillAboveMidSellsBelow) {
    ExecutionModel model(0.001, 0.01, 0.0);
    EXPECT_DOUBLE_EQ(model.fill_price(100.0, 5.0, 0.5), 100.0 * (1.0 + 0.001 + 0.01 * 0.5));
    EXPECT_DOUBLE_EQ(model.fill_price(100.0, -5.0, 0.5), 100.0 * (1.0 - 0.001 - 0.01 * 0.5));
    EXPECT_DOUBLE_EQ(model.fill_price(100.0, 0.0, 0.0), 100.0);
}

TEST(ExecutionModel, CommissionIsRateTimesNotional) {
    ExecutionModel model(0.0, 0.0, 0.01);
    EXPECT_DOUBLE_EQ(model.commission(3.0, 99.0), 0.01 * 3.0 * 99.0);
    EXPECT_DOUBLE_EQ(model.commission(-3.0, 99.0), 0.01 * 3.0 * 99.0);
    EXPECT_DOUBLE_EQ(model.commission(0.0, 99.0), 0.0);
}

TEST(ExecutionModel, RejectsInvalidParameters) {
    EXPECT_THROW(ExecutionModel(-0.1, 0.0, 0.0), ConfigError);
    EXPECT_THROW(ExecutionModel(1.0, 0.0, 0.0), ConfigError);
    EXPECT_THROW(ExecutionModel(0.0, -1.0, 0.0), ConfigError);
    EXPECT_THROW(ExecutionModel(0.0, 0.0, -0.01), ConfigError);
    EXPECT_THROW(ExecutionModel(0.0, 0.0, 1.0), ConfigError);
}

TEST(Env, CommissionChargedInCurrency) {
    // Regression test: a 1% commission on a ~$10,000 fill must cost ~$100, not ~$0.01.
    auto env = make_env(make_data({100, 100, 100}, {100, 100, 100}), costs(0.0, 0.0, 0.01));
    env.reset();
    auto r = env.step(1.0);
    const double notional = r.info.units_traded * r.info.fill_price;
    EXPECT_NEAR(notional, 10000.0, 1e-9);
    EXPECT_NEAR(r.info.commission, 100.0, 1e-9);
    EXPECT_NEAR(r.info.equity, 10000.0 - 100.0, 1e-9);
}

TEST(Env, SpreadMovesFillAgainstTheTrader) {
    auto env = make_env(make_data({100, 100, 100, 100}, {100, 100, 100, 100}), costs(0.001, 0.0, 0.0));
    env.reset();
    auto buy = env.step(1.0);
    EXPECT_GT(buy.info.fill_price, 100.0);
    auto sell = env.step(-1.0);
    EXPECT_LT(sell.info.fill_price, 100.0);
    EXPECT_LT(sell.info.equity, 10000.0);
}

TEST(Env, ImpactGrowsWithTradedExposure) {
    auto small = make_env(make_data({100, 100, 100}, {100, 100, 100}), costs(0.0, 0.01, 0.0));
    auto large = make_env(make_data({100, 100, 100}, {100, 100, 100}), costs(0.0, 0.01, 0.0));
    small.reset();
    large.reset();
    const double small_fill = small.step(0.25).info.fill_price;
    const double large_fill = large.step(1.0).info.fill_price;
    EXPECT_GT(large_fill, small_fill);
    EXPECT_NEAR(large_fill, 100.0 * (1.0 + 0.01 * 1.0), 1e-6);  // impact on |delta exposure| = 1
}

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ExecutionModel.h"
#include "MarketData.h"
#include "RewardFunctions.h"
#include "TradingEnv.h"

namespace test {

using Columns = std::vector<std::pair<std::string, std::vector<double>>>;

/// Builds MarketData with "open"/"close" columns plus any extra columns.
inline std::shared_ptr<const MarketData> make_data(
    const std::vector<double>& open, const std::vector<double>& close, Columns extra = {}) {
    std::vector<std::int64_t> ts(open.size());
    for (std::size_t i = 0; i < ts.size(); ++i) {
        ts[i] = static_cast<std::int64_t>(i) * 3600;
    }
    Columns columns{{"open", open}, {"close", close}};
    for (auto& column : extra) {
        columns.push_back(std::move(column));
    }
    return std::make_shared<const MarketData>(MarketData::from_columns(std::move(ts), columns));
}

inline std::shared_ptr<const ExecutionModel> costs(double half_spread = 0.0, double impact = 0.0,
                                                   double commission = 0.0) {
    return std::make_shared<const ExecutionModel>(half_spread, impact, commission);
}

inline TradingEnv make_env(std::shared_ptr<const MarketData> data,
                           std::shared_ptr<const ExecutionModel> execution = costs(),
                           EnvConfig config = {}) {
    return TradingEnv(std::move(data), std::move(execution), RewardFunctions::log_return_reward(),
                      std::move(config));
}

}  // namespace test

#endif

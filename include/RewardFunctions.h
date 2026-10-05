#ifndef REWARD_FUNCTIONS_H
#define REWARD_FUNCTIONS_H

#include <functional>

#include "Bar.h"
#include "Portfolio.h"

/// @file RewardFunctions.h
/// @brief Defines the objective signal for the RL agent.

namespace RewardFunctions {

/// @brief Smallest equity ratio used inside the log, so a bankrupting step yields a large
/// but finite penalty (log(1e-6) ~ -13.8) instead of NaN.
constexpr double MIN_EQUITY_RATIO = 1e-6;

/// @brief Function signature for reward calculations.
/// @param previous Portfolio before the step (marked at close t).
/// @param current Portfolio after the step (marked at close t+1).
/// @param current_bar Bar t+1.
using RewardFn = std::function<double(
    const Portfolio& previous,
    const Portfolio& current,
    const Bar& current_bar
)>;

/// @brief Log growth of equity over the step: log(equity_{t+1} / equity_t), floored.
/// Summing it over an episode gives the episode's log return net of all costs.
/// @return RewardFn.
RewardFn log_return_reward();

}  // namespace RewardFunctions

#endif

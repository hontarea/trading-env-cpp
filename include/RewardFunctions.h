#ifndef REWARD_FUNCTIONS_H
#define REWARD_FUNCTIONS_H

#include <functional>

#include "Bar.h"
#include "Portfolio.h"

/// @file RewardFunctions.h
/// @brief Defines the objective signal for the RL agent.

namespace RewardFunctions {

/// @brief Function signature for reward calculations.
using RewardFn = std::function<double(
    const Portfolio& previous,
    const Portfolio& current,
    const Bar& current_bar
)>;

/// @brief Creates a log-return based reward function.
/// @return RewardFn.
RewardFn log_return_reward();

} 

#endif 

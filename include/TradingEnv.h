/// @file TradingEnv.h
/// @brief Implements the Gymnasium API bridge for RL training.

#ifndef TRADING_ENV_H
#define TRADING_ENV_H

#include <memory>

#include "ExecutionModel.h"
#include "MarketData.h"
#include "Portfolio.h"
#include "RewardFunctions.h"

// Trading Environment Constants
constexpr double DEFAULT_INITIAL_CASH = 10000.0;
constexpr std::size_t DEFAULT_START_INDEX = 0;
constexpr std::size_t DEFAULT_END_INDEX = 0;
constexpr std::size_t SMA_WINDOW_SIZE = 5;

// Exception Messages
constexpr char EXCEPTION_DATA_NULL_EMPTY[] = "MarketData must be non-null and non-empty";
constexpr char EXCEPTION_EXEC_NULL[] = "ExecutionModel must be non-null";
constexpr char EXCEPTION_REWARD_NULL[] = "Reward function must be non-null";
constexpr char EXCEPTION_START_GTE_END[] = "start_index ({}) must be less than end_index ({})";
constexpr char EXCEPTION_END_EXCEEDS_DATA[] = "end_index ({}) exceeds data size ({})";
constexpr char EXCEPTION_DATA_TOO_SHORT[] = "Data range must contain at least 2 bars";
constexpr char EXCEPTION_INVALID_ACTION[] = "Action must be in {{-1, 0, +1}}, got {}";

/// @brief Configuration settings for TradingEnv.
/// Decouples environment construction from simulation hyperparameters.
struct EnvConfig {
    double initial_cash = DEFAULT_INITIAL_CASH;    
    std::size_t start_index = DEFAULT_START_INDEX;      
    std::size_t end_index = DEFAULT_END_INDEX;       
};

/// @brief Observations of the environment at a single step.
/// Exposes critical features for the agent's observation space.
struct Observation {
    double last_log_return = 0.0;          
    double short_window_mean_return = 0.0; 
    int current_position = 0;              
    double equity = 0.0;
    double cash = 0.0;
    int steps_remaining = 0;
    double forecast = 0.0;
};

/// @brief The outcome of taking a step in the environment.
/// Mirrors Gymnasium's step result tuple format.
struct StepResult {
    Observation observation;
    double reward = 0.0;     
    bool terminated = false; 
    bool truncated = false;  
};

/// @brief The primary interface for the trading simulation.
/// Encapsulates data, execution, and state progression for RL.
class TradingEnv {
public:
    /// @brief Constructs the Trading Environment.
    /// @param data Shared pointer to historical data.
    /// @param execution Shared pointer to execution model.
    /// @param reward_fn The function to calculate step rewards.
    /// @param config Starting configuration values.
    TradingEnv(
        std::shared_ptr<const MarketData> data,
        std::shared_ptr<ExecutionModel> execution,
        RewardFunctions::RewardFn reward_fn,
        EnvConfig config);

    /// @brief Resets the environment to its initial state.
    /// Required to start a new episode.
    /// @return The initial observation.
    Observation reset();

    /// @brief Advances the environment by taking an action.
    /// Allows the agent to interact and receive feedback.
    /// @param action The desired position state (-1, 0, +1).
    /// @return The resulting state and reward.
    StepResult step(int action);

    /// @brief Checks if the episode is finished.
    /// @return True if done.
    bool done() const;

    /// @brief Accesses the current portfolio state.
    /// @return Reference to the Portfolio.
    const Portfolio& portfolio() const;

    /// @brief Retrieves the current step index.
    /// @return The internal step number.
    std::size_t current_step() const;

private:
    /// @brief Generates an observation from the current state.
    /// @return Formatted Observation struct.
    Observation make_observation() const;

    std::shared_ptr<const MarketData> data_; 
    std::shared_ptr<ExecutionModel> execution_; 
    RewardFunctions::RewardFn reward_fn_; 
    Portfolio portfolio_; 
    std::size_t current_step_ = 0; 
    double initial_cash_; 
    std::size_t start_index_;
    std::size_t end_index_;

    // Indices for critical features
    size_t open_idx_;
    size_t close_idx_;
    size_t forecast_idx_;
};

#endif 

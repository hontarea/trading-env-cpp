/// @file TradingEnv.h
/// @brief Single-asset trading environment with a Gymnasium-style reset/step API.

#ifndef TRADING_ENV_H
#define TRADING_ENV_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ExecutionModel.h"
#include "MarketData.h"
#include "Portfolio.h"
#include "RewardFunctions.h"

// Trading Environment Constants
constexpr double DEFAULT_INITIAL_CASH = 10000.0;
constexpr std::size_t DEFAULT_START_INDEX = 0;
constexpr std::size_t DEFAULT_END_INDEX = 0;   ///< 0 means "end of data".
constexpr double DEFAULT_MAX_LEVERAGE = 1.0;
constexpr std::size_t DEFAULT_OBS_LOOKBACK = 1;
constexpr std::size_t SMA_WINDOW_SIZE = 5;

// Exception Messages
constexpr char EXCEPTION_DATA_NULL_EMPTY[] = "MarketData must be non-null and non-empty";
constexpr char EXCEPTION_EXEC_NULL[] = "ExecutionModel must be non-null";
constexpr char EXCEPTION_REWARD_NULL[] = "Reward function must be non-null";
constexpr char EXCEPTION_INITIAL_CASH[] = "initial_cash must be positive and finite, got {}";
constexpr char EXCEPTION_MAX_LEVERAGE[] = "max_leverage must be positive and finite, got {}";
constexpr char EXCEPTION_OBS_LOOKBACK[] = "obs_lookback must be >= 1";
constexpr char EXCEPTION_RANGE[] = "Episode range [{}, {}) is invalid for data of size {}: need start < end <= size and at least 2 bars";
constexpr char EXCEPTION_NON_POSITIVE_PRICE[] = "Bar {} has a non-positive '{}' price ({})";
constexpr char EXCEPTION_INVALID_ACTION[] = "target_exposure must be finite and in [-1, 1], got {}";
constexpr char EXCEPTION_NOT_RUNNING[] = "step() called without an active episode; call reset() first";

/// @brief Configuration settings for TradingEnv.
/// Decouples environment construction from simulation hyperparameters.
struct EnvConfig {
    double initial_cash = DEFAULT_INITIAL_CASH;
    std::size_t start_index = DEFAULT_START_INDEX;  ///< First bar of the default episode.
    std::size_t end_index = DEFAULT_END_INDEX;      ///< One past the last bar; 0 = data size.
    double max_leverage = DEFAULT_MAX_LEVERAGE;     ///< |exposure| reached by action +-1.
    std::vector<std::string> obs_features;          ///< Extra data columns in the observation.
    std::size_t obs_lookback = DEFAULT_OBS_LOOKBACK; ///< Bars of history per obs feature.
};

/// @brief Per-step execution details, mirrored into Gymnasium's `info` dict.
struct StepInfo {
    double fill_price = 0.0;    ///< Execution price (mid price if nothing was traded).
    double units_traded = 0.0;  ///< Signed units bought (+) or sold (-).
    double commission = 0.0;    ///< Commission paid, in currency.
    double equity = 0.0;        ///< Equity after marking at close t+1.
    double exposure = 0.0;      ///< Signed position value / equity after the step.
    bool bankrupt = false;      ///< Equity fell to <= 0.
};

/// @brief The outcome of taking a step in the environment.
/// Mirrors Gymnasium's (obs, reward, terminated, truncated, info) tuple.
struct StepResult {
    std::vector<double> observation;
    double reward = 0.0;
    bool terminated = false;  ///< The MDP ended (bankruptcy).
    bool truncated = false;   ///< The episode's data range ran out; observation is still valid.
    StepInfo info;
};

/// @brief Single-asset trading environment.
///
/// Step timeline (no look-ahead): the observation at step t uses only bars <= t
/// (its close and features). The action is an order executed at the *open* of bar t+1,
/// and the reward marks the portfolio to the *close* of bar t+1.
///
/// Action: target exposure a in [-1, 1]. At open[t+1] the position is rebalanced to
///   units = a * max_leverage * equity(open[t+1]) / open[t+1].
///
/// Observation (see observation_names()):
///   market part   - last log return, mean log return over SMA_WINDOW_SIZE bars, then each
///                   obs_features column at t, t-1, ..., t-obs_lookback+1;
///   portfolio part - exposure, log(equity / episode start equity), fraction of episode left.
class TradingEnv {
public:
    /// @brief Number of portfolio-state entries at the end of every observation.
    static constexpr std::size_t PORTFOLIO_FEATURES = 3;

    /// @brief Constructs the Trading Environment.
    /// @param data Shared pointer to historical data.
    /// @param execution Shared pointer to execution model.
    /// @param reward_fn The function to calculate step rewards.
    /// @param config Starting configuration values.
    /// @throws ConfigError on invalid arguments.
    TradingEnv(
        std::shared_ptr<const MarketData> data,
        std::shared_ptr<const ExecutionModel> execution,
        RewardFunctions::RewardFn reward_fn,
        EnvConfig config);

    /// @brief Starts an episode over the configured [start_index, end_index) range.
    /// @return The initial observation.
    std::vector<double> reset();

    /// @brief Starts an episode over the bar range [start_index, end_index).
    /// @throws ConfigError if the range is invalid.
    std::vector<double> reset(std::size_t start_index, std::size_t end_index);

    /// @brief Executes one step.
    /// @param target_exposure Desired exposure in [-1, 1] (scaled by max_leverage).
    /// @throws InvalidActionError if the action is out of range.
    /// @throws EpisodeError if no episode is active.
    StepResult step(double target_exposure);

    /// @brief True when no episode is active (before reset() or after it ended).
    bool done() const;

    /// @brief Accesses the current portfolio state.
    const Portfolio& portfolio() const;

    /// @brief Index of the current bar.
    std::size_t current_step() const;

    std::size_t episode_start() const { return episode_start_; }
    std::size_t episode_end() const { return episode_end_; }

    /// @brief Total observation length.
    std::size_t observation_size() const;

    /// @brief Human-readable names of the observation entries, in order.
    std::vector<std::string> observation_names() const;

    /// @brief Number of leading observation entries that depend only on market data.
    std::size_t market_feature_count() const;

    /// @brief The market part of the observation at bar t (depends only on bars <= t).
    /// Useful for fitting normalization statistics on a training range.
    std::vector<double> market_features(std::size_t t) const;

    const EnvConfig& config() const { return config_; }

private:
    /// @brief Builds the observation for the current state.
    std::vector<double> make_observation() const;

    double open_at(std::size_t t) const { return (*data_)[t].get_feature(open_idx_); }
    double close_at(std::size_t t) const { return (*data_)[t].get_feature(close_idx_); }

    std::shared_ptr<const MarketData> data_;
    std::shared_ptr<const ExecutionModel> execution_;
    RewardFunctions::RewardFn reward_fn_;
    EnvConfig config_;
    Portfolio portfolio_;

    std::size_t current_step_ = 0;
    std::size_t episode_start_ = 0;
    std::size_t episode_end_ = 0;
    double episode_start_equity_ = 0.0;
    bool running_ = false;

    // Indices for critical features
    std::size_t open_idx_ = 0;
    std::size_t close_idx_ = 0;
    std::vector<std::size_t> obs_feature_idx_;
};

#endif

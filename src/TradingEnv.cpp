#include <algorithm>
#include <cmath>
#include <format>

#include "Exceptions.h"
#include "ExecutionModel.h"
#include "TradingEnv.h"

namespace {

void validate_range(std::size_t start, std::size_t end, std::size_t size) {
    if (start >= end || end > size || end - start < 2) {
        throw ConfigError(std::format(EXCEPTION_RANGE, start, end, size));
    }
}

}  // namespace

TradingEnv::TradingEnv(
    std::shared_ptr<const MarketData> data,
    std::shared_ptr<const ExecutionModel> execution,
    RewardFunctions::RewardFn reward_fn,
    EnvConfig config)
    : data_(std::move(data)),
      execution_(std::move(execution)),
      reward_fn_(std::move(reward_fn)),
      config_(std::move(config)),
      portfolio_(config_.initial_cash) {
    // Check the validity of the environment's params
    if (!data_ || data_->size() == 0) {
        throw ConfigError(EXCEPTION_DATA_NULL_EMPTY);
    }
    if (!execution_) {
        throw ConfigError(EXCEPTION_EXEC_NULL);
    }
    if (!reward_fn_) {
        throw ConfigError(EXCEPTION_REWARD_NULL);
    }
    if (!(config_.initial_cash > 0.0) || !std::isfinite(config_.initial_cash)) {
        throw ConfigError(std::format(EXCEPTION_INITIAL_CASH, config_.initial_cash));
    }
    if (!(config_.max_leverage > 0.0) || !std::isfinite(config_.max_leverage)) {
        throw ConfigError(std::format(EXCEPTION_MAX_LEVERAGE, config_.max_leverage));
    }
    if (config_.obs_lookback < 1) {
        throw ConfigError(EXCEPTION_OBS_LOOKBACK);
    }
    if (config_.end_index == DEFAULT_END_INDEX) {
        config_.end_index = data_->size();
    }
    validate_range(config_.start_index, config_.end_index, data_->size());

    open_idx_ = data_->get_feature_index(FEATURE_OPEN);
    close_idx_ = data_->get_feature_index(FEATURE_CLOSE);
    for (const auto& name : config_.obs_features) {
        obs_feature_idx_.push_back(data_->get_feature_index(name));
    }

    // Prices feed log() and position sizing, so they must be strictly positive.
    for (std::size_t t = 0; t < data_->size(); ++t) {
        for (auto [idx, name] : {std::pair{open_idx_, FEATURE_OPEN}, std::pair{close_idx_, FEATURE_CLOSE}}) {
            const double price = (*data_)[t].get_feature(idx);
            if (!(price > 0.0)) {
                throw ConfigError(std::format(EXCEPTION_NON_POSITIVE_PRICE, t, name, price));
            }
        }
    }
}

std::vector<double> TradingEnv::reset() {
    return reset(config_.start_index, config_.end_index);
}

std::vector<double> TradingEnv::reset(std::size_t start_index, std::size_t end_index) {
    validate_range(start_index, end_index, data_->size());
    episode_start_ = start_index;
    episode_end_ = end_index;
    current_step_ = start_index;
    portfolio_ = Portfolio(config_.initial_cash);
    portfolio_.mark_to_market(close_at(current_step_));
    episode_start_equity_ = portfolio_.equity();
    running_ = true;
    return make_observation();
}

StepResult TradingEnv::step(double target_exposure) {
    if (!running_) {
        throw EpisodeError(EXCEPTION_NOT_RUNNING);
    }
    if (!std::isfinite(target_exposure) || std::abs(target_exposure) > 1.0) {
        throw InvalidActionError(std::format(EXCEPTION_INVALID_ACTION, target_exposure));
    }

    const Portfolio previous = portfolio_;
    const std::size_t next = current_step_ + 1;
    const double mid_price = open_at(next);

    // Size the order against equity valued at the execution price. If an overnight gap has
    // already wiped out equity, the only sensible order is to close the position.
    const double equity_at_open = portfolio_.equity_at(mid_price);
    const double target_units = equity_at_open > 0.0
        ? target_exposure * config_.max_leverage * equity_at_open / mid_price
        : 0.0;
    const double units_delta = target_units - portfolio_.units();
    const double delta_exposure = equity_at_open > 0.0
        ? std::abs(units_delta) * mid_price / equity_at_open
        : 0.0;

    StepResult result;
    result.info.fill_price = execution_->fill_price(mid_price, units_delta, delta_exposure);
    if (units_delta != 0.0) {
        result.info.commission = execution_->commission(units_delta, result.info.fill_price);
        result.info.units_traded = units_delta;
        portfolio_.apply_fill(units_delta, result.info.fill_price, result.info.commission);
    }

    current_step_ = next;
    portfolio_.mark_to_market(close_at(current_step_));

    result.reward = reward_fn_(previous, portfolio_, (*data_)[current_step_]);
    result.info.equity = portfolio_.equity();
    result.info.exposure = portfolio_.exposure();
    result.info.bankrupt = portfolio_.equity() <= 0.0;

    // Bankruptcy ends the MDP; running out of bars is a time limit, not a terminal state.
    result.terminated = result.info.bankrupt;
    result.truncated = !result.terminated && current_step_ + 1 >= episode_end_;
    running_ = !(result.terminated || result.truncated);

    result.observation = make_observation();
    return result;
}

bool TradingEnv::done() const {
    return !running_;
}

const Portfolio& TradingEnv::portfolio() const {
    return portfolio_;
}

std::size_t TradingEnv::current_step() const {
    return current_step_;
}

std::size_t TradingEnv::market_feature_count() const {
    return 2 + obs_feature_idx_.size() * config_.obs_lookback;
}

std::size_t TradingEnv::observation_size() const {
    return market_feature_count() + PORTFOLIO_FEATURES;
}

std::vector<std::string> TradingEnv::observation_names() const {
    std::vector<std::string> names{"log_return", std::format("mean_log_return_{}", SMA_WINDOW_SIZE)};
    for (const auto& feature : config_.obs_features) {
        for (std::size_t lag = 0; lag < config_.obs_lookback; ++lag) {
            names.push_back(lag == 0 ? feature : std::format("{}_lag{}", feature, lag));
        }
    }
    names.insert(names.end(), {"exposure", "log_equity_growth", "episode_frac_remaining"});
    return names;
}

std::vector<double> TradingEnv::market_features(std::size_t t) const {
    const Bar& bar = data_->at(t);
    std::vector<double> features;
    features.reserve(market_feature_count());

    features.push_back(t >= 1 ? log_return((*data_)[t - 1], bar, close_idx_) : 0.0);

    // Mean of the last SMA_WINDOW_SIZE close-to-close log returns (fewer near the data start).
    const std::size_t window = std::min(SMA_WINDOW_SIZE, t);
    features.push_back(window > 0
        ? std::log(close_at(t) / close_at(t - window)) / static_cast<double>(window)
        : 0.0);

    // Lags before the first bar repeat bar 0: still only past information.
    for (std::size_t idx : obs_feature_idx_) {
        for (std::size_t lag = 0; lag < config_.obs_lookback; ++lag) {
            features.push_back((*data_)[t >= lag ? t - lag : 0].get_feature(idx));
        }
    }
    return features;
}

std::vector<double> TradingEnv::make_observation() const {
    std::vector<double> obs = market_features(current_step_);
    const double equity_ratio = portfolio_.equity() / episode_start_equity_;
    const double steps_total = static_cast<double>(episode_end_ - 1 - episode_start_);
    const double steps_left = static_cast<double>(episode_end_ - 1 - current_step_);

    obs.push_back(portfolio_.exposure());
    obs.push_back(std::log(std::max(equity_ratio, RewardFunctions::MIN_EQUITY_RATIO)));
    obs.push_back(steps_left / steps_total);
    return obs;
}

#include <cmath>
#include <format>

#include "Exceptions.h"
#include "TradingEnv.h"

TradingEnv::TradingEnv(
    std::shared_ptr<const MarketData> data,
    std::shared_ptr<ExecutionModel> execution,
    RewardFunctions::RewardFn reward_fn,
    EnvConfig config)
: data_(std::move(data)),
      execution_(std::move(execution)),
      reward_fn_(std::move(reward_fn)),
      portfolio_(config.initial_cash),
      initial_cash_(config.initial_cash),
      start_index_(config.start_index),
      end_index_(config.end_index == 0 ? data_->size() : config.end_index) {
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
    if (start_index_ >= end_index_) {
        throw ConfigError(std::format(EXCEPTION_START_GTE_END, start_index_, end_index_));
    }
    if (end_index_ > data_->size()) {
        throw ConfigError(std::format(EXCEPTION_END_EXCEEDS_DATA, end_index_, data_->size()));
    }
    if (end_index_ - start_index_ < 2) {
        throw ConfigError(EXCEPTION_DATA_TOO_SHORT);
    }

    open_idx_ = data_->get_feature_index(FEATURE_OPEN);
    close_idx_ = data_->get_feature_index(FEATURE_CLOSE);
    forecast_idx_ = data_->get_feature_index(FEATURE_FORECAST);
}

Observation TradingEnv::reset() {
    portfolio_ = Portfolio(initial_cash_);
    current_step_ = start_index_;
    portfolio_.update_equity((*data_)[current_step_].get_feature(close_idx_));
    return make_observation();
}

StepResult TradingEnv::step(int action) {
    if (action < -1 || action > 1) {
        throw ConfigError(std::format(EXCEPTION_INVALID_ACTION, action));
    }

    if (current_step_ + 1 >= end_index_) {
        return StepResult{Observation{}, 0.0, true, false};
    }

    Portfolio previous = portfolio_;

    const Bar& next_bar = (*data_)[current_step_ + 1];
    double mid_price = next_bar.get_feature(open_idx_);

    int position_change = action - portfolio_.position();
    double fill_price = execution_->compute_fill_price(mid_price, position_change);
    double txn_cost_log = execution_->compute_transaction_cost_log(position_change);

    portfolio_.apply_fill(action, fill_price, txn_cost_log);
    portfolio_.step_forward();

    ++current_step_;

    bool terminated = (current_step_ + 1 >= end_index_);

    portfolio_.update_equity(next_bar.get_feature(close_idx_));

    double reward = reward_fn_(previous, portfolio_, next_bar);

    Observation obs = terminated ? Observation{} : make_observation();

    return StepResult{obs, reward, terminated, false};
}

bool TradingEnv::done() const {
    return current_step_ + 1 >= end_index_;
}

const Portfolio& TradingEnv::portfolio() const {
    return portfolio_;
}

std::size_t TradingEnv::current_step() const  {
    return current_step_;
}


Observation TradingEnv::make_observation() const {
    Observation obs;
    obs.current_position = portfolio_.position();
    obs.equity = portfolio_.equity();
    obs.cash = portfolio_.cash();
    obs.steps_remaining = static_cast<int>(end_index_ - current_step_);
    obs.forecast = (*data_)[current_step_].get_feature(forecast_idx_);

    if (current_step_ >= 1) {
        obs.last_log_return = log_return(
            (*data_)[current_step_ - 1], (*data_)[current_step_], close_idx_);
    }

    std::size_t window = SMA_WINDOW_SIZE;
    std::size_t available = current_step_ - start_index_;
    if (available < window) {
        window = available;
    }
    if (window > 0) {
        double sum = 0.0;
        for (std::size_t i = 0; i < window; ++i) {
            std::size_t idx = current_step_ - i;
            sum += log_return((*data_)[idx - 1], (*data_)[idx], close_idx_);
        }
        obs.short_window_mean_return = sum / static_cast<double>(window);
    }

    return obs;
}

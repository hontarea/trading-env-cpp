#include <cmath>
#include <format>

#include "Exceptions.h"
#include "ExecutionModel.h"

ExecutionModel::ExecutionModel(double half_spread, double impact_coef, double txn_cost_rate)
    : half_spread_(half_spread), impact_coef_(impact_coef), txn_cost_rate_(txn_cost_rate) {
    if (half_spread_ < 0.0) {
        throw ConfigError(std::format(EXCEPTION_HALF_SPREAD_NEG, half_spread_));
    }
    if (impact_coef_ < 0.0) {
        throw ConfigError(std::format(EXCEPTION_IMPACT_COEF_NEG, impact_coef_));
    }
    if (txn_cost_rate_ < 0.0) {
        throw ConfigError(std::format(EXCEPTION_TXN_COST_RATE_NEG, txn_cost_rate_));
    }
    if (txn_cost_rate_ * MAX_POSITION_CHANGE >= 1.0) {
        throw ConfigError(std::format(EXCEPTION_TXN_COST_RATE_TOO_HIGH, MAX_POSITION_CHANGE, MAX_POSITION_CHANGE, txn_cost_rate_));
    }
}

double ExecutionModel::compute_fill_price(double mid_price, int position_change) const {
    if (position_change == POSITION_FLAT) {
        return mid_price;
    }
    double sign = (position_change > 0) ? OPEN_LONG : OPEN_SHORT;
    double abs_change = static_cast<double>(std::abs(position_change));
    return mid_price * (1.0 + sign * (half_spread_ + impact_coef_ * abs_change));
}

double ExecutionModel::compute_transaction_cost_log(int position_change) const {
    if (position_change == POSITION_FLAT) {
        return 0.0;
    }
    double abs_change = static_cast<double>(std::abs(position_change));
    return -std::log(1.0 - txn_cost_rate_ * abs_change);
}
